from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import math

# Use the correct path for template folder structure in common environments
app = Flask(__name__, template_folder='client/templates')

# --- Load your saved model & encoder (paths used in your Colab)
MODEL_PATH = "post_analysis_proxy_model.pkl"
ENCODER_PATH = "post_type_encoder.pkl"

# Placeholder loading (assuming the user has these files)
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
except FileNotFoundError:
    print("Warning: Model or encoder files not found. Using dummy data for analysis endpoints.")
    # Implement dummy model/encoder for local testing without the ML files
    class DummyEncoder:
        def classes_(self): return ["Reel", "Video", "Image", "Story"]
        def transform(self, pt): 
            mapping = {"Reel": 0, "Video": 1, "Image": 2, "Story": 3}
            return [mapping.get(pt[0], 0)]
    class DummyModel:
        def predict(self, df): return [2500]
    model = DummyModel()
    encoder = DummyEncoder()


# --- helper mappings
# Best time suggestions by post type (you can tune)
BEST_TIME = {
    "Reel": "6:00 PM - 9:00 PM",
    "Video": "7:00 PM - 10:00 PM",
    "Image": "5:00 PM - 8:00 PM",
    "Story": "12:00 PM - 2:00 PM"
}

# simple hashtag suggestions per post type
HASHTAGS = {
    "Reel": ["#reels", "#viral", "#trending", "#reelitfeelit"],
    "Video": ["#video", "#watchthis", "#instavideo"],
    "Image": ["#photooftheday", "#instagood", "#snapshot"],
    "Story": ["#storytime", "#behindthescenes", "#daily"]
}

POSITIVE_WORDS = set(["love","great","awesome","amazing","good","best","nice","happy","fantastic","excellent"])
NEGATIVE_WORDS = set(["hate","terrible","bad","awful","worst","sad","angry","disappointing","problem","fail"])

def normalize_post_type(pt):
    return pt.strip().capitalize()

def compute_engagement_score(likes, comments, shares, saves, predicted_impr):
    # engagement rate as percentage of predicted impressions
    total_engagement = likes + comments + shares + saves
    denom = max(predicted_impr, 1)
    score = (total_engagement / denom) * 100
    # cap between 0 and 100
    score = max(0.0, min(score, 100.0))
    return round(score, 2)

def sentiment_proxy(text):
    # a very small rule-based sentiment proxy (no external deps)
    if not text or str(text).strip()=="":
        return {"sentiment":"N/A", "score": None}
    text_l = str(text).lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_l)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_l)
    if pos==0 and neg==0:
        return {"sentiment":"Neutral", "score":0}
    if pos >= neg:
        return {"sentiment":"Positive", "score": pos - neg}
    else:
        return {"sentiment":"Negative", "score": neg - pos}

def tips_and_actions(post_type, engagement_score, sentiment):
    tips = []
    # baseline suggestions
    if engagement_score < 2:
        tips.append("Low engagement predicted — add a clear call-to-action (ask a question), use trending audio, and promote with stories.")
    elif engagement_score < 8:
        tips.append("Moderate engagement — try shortening caption, add 2-3 trending hashtags, and post at peak hours.")
    else:
        tips.append("Good engagement — keep the format and posting time. Consider boosting this post for wider reach.")
    # post-type specific
    if post_type == "Reel":
        tips.append("Reels benefit from trending audio and vertical framing. Use first 2 seconds to hook viewers.")
    elif post_type == "Image":
        tips.append("Use high-quality visuals & descriptive captions. Carousel posts often increase saves.")
    elif post_type == "Video":
        tips.append("Add captions/subtitles and an engaging thumbnail to improve clicks.")
    elif post_type == "Story":
        tips.append("Use interactive stickers (polls/ques) to increase replies and retention.")
    # sentiment-based tips
    if sentiment["sentiment"] == "Negative":
        tips.append("Caption sentiment is negative — consider rephrasing to be more solution-focused or add empathy.")
    return tips

# --- Updated: Serve the single index.html for all front-end routes
@app.route('/')
@app.route('/analyze')
@app.route('/about')
@app.route('/contact')
def serve_index():
    """Serves the main single-page application (SPA) file."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    json_in = request.get_json()
    if not json_in:
        return jsonify({"error":"Invalid or empty JSON"}), 400

    try:
        # read inputs (required fields)
        post_type = str(json_in.get("Post_Type", "")).strip()
        caption = json_in.get("Caption", "") # optional
        try:
            likes = int(json_in.get("Likes", 0))
            comments = int(json_in.get("Comments", 0))
            shares = int(json_in.get("Shares", 0))
            saves = int(json_in.get("Saves", 0))
        except Exception:
            return jsonify({"error":"Likes/Comments/Shares/Saves must be integers"}), 400

        # normalize and validate post type
        post_type_clean = normalize_post_type(post_type)
        allowed = list(encoder.classes_)
        if post_type_clean not in allowed:
            return jsonify({"error": f"Invalid Post_Type '{post_type}'. Allowed: {allowed}"}), 400

        # encode and predict
        encoded = encoder.transform([post_type_clean])[0]
        input_df = pd.DataFrame({
            # NOTE: The prediction model likely expects 'Post_Type' to be the 
            # only encoded feature (e.g., [0]), and the engagement metrics 
            # (Likes, Comments, Shares, Saves) are the known inputs.
            "Post_Type":[encoded],
            "Likes":[likes],
            "Comments":[comments],
            "Shares":[shares],
            "Saves":[saves]
        })
        
        # Handle the case where the model is a dummy and doesn't support predict
        try:
            predicted_impr = float(model.predict(input_df)[0])
        except AttributeError:
            # Fallback for dummy model
            predicted_impr = model.predict(input_df)[0]
            
        predicted_impr_int = int(round(predicted_impr))

        # compute engagement score (based on predicted impressions)
        engagement = compute_engagement_score(likes, comments, shares, saves, predicted_impr)

        # sentiment proxy for caption
        sentiment = sentiment_proxy(caption)

        # best posting time + hashtags + tips
        best_time = BEST_TIME.get(post_type_clean, "Anytime")
        hashtags = HASHTAGS.get(post_type_clean, ["#post"])
        tips = tips_and_actions(post_type_clean, engagement, sentiment)

        response = {
            "Predicted_Impressions": predicted_impr_int,
            "Predicted_Impressions_raw": predicted_impr,
            "Engagement_Score_pct": engagement,
            "Best_Time_To_Post": best_time,
            "Suggested_Hashtags": hashtags,
            "Tips": tips,
            "Caption_Sentiment": sentiment
        }
        return jsonify(response)

    except Exception as e:
        # CRITICAL: Catch any unhandled internal errors (e.g., ML model issue, DataFrame problem)
        print(f"CRITICAL SERVER ERROR during prediction: {e}")
        # Return a generic 500 error with a helpful message for the user
        return jsonify({
            "error": f"Internal Server Error during analysis ({type(e).__name__}). Please check the server logs for the full traceback."
        }), 500


if __name__ == "__main__":
    # You might need to change the template_folder path for Colab depending on your file structure
    app.run(host="0.0.0.0", port=5000, debug=True)
