import streamlit as st
import cv2
import numpy as np
import joblib

from feature_extraction import extract_features


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Facial Stress Classification",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Facial Expression-Based Stress Classification")

st.write(
    "Real-time facial behavioural analysis and "
    "stress-risk estimation using Machine Learning."
)

st.warning(
    "This application provides an experimental "
    "stress-risk estimate and is not a medical diagnosis."
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load("stress_model.pkl")

except FileNotFoundError:

    st.error(
        "stress_model.pkl was not found."
    )

    st.info(
        "Make sure stress_model.pkl is in the same "
        "folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Error loading stress_model.pkl: {e}"
    )

    st.stop()


# ============================================================
# GET EXPECTED NUMBER OF FEATURES
# ============================================================

try:

    expected_features = model.n_features_in_

except AttributeError:

    expected_features = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Controls")

run_camera = st.sidebar.checkbox(
    "📷 Start Camera"
)

if expected_features is not None:

    st.sidebar.write(
        f"Model expects: {expected_features} features"
    )


# ============================================================
# CAMERA START
# ============================================================

if run_camera:

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        st.error(
            "❌ Unable to access the camera."
        )

        st.info(
            "Check that your webcam is connected "
            "and not being used by another application."
        )

        st.stop()


    # Streamlit placeholders

    frame_placeholder = st.empty()

    prediction_placeholder = st.empty()


    try:

        while True:

            # ==================================================
            # READ CAMERA FRAME
            # ==================================================

            success, frame = camera.read()

            if not success:

                st.error(
                    "Unable to read frame from camera."
                )

                break


            # ==================================================
            # MIRROR CAMERA
            # ==================================================

            frame = cv2.flip(
                frame,
                1
            )


            # ==================================================
            # EXTRACT FEATURES
            # ==================================================

            try:

                features = extract_features(
                    frame
                )

            except Exception as e:

                st.error(
                    f"Feature extraction error: {e}"
                )

                break


            # ==================================================
            # NO FACE DETECTED
            # ==================================================

            if features is None:

                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                prediction_placeholder.warning(
                    "No face detected. "
                    "Please look directly at the camera."
                )


            # ==================================================
            # FACE DETECTED
            # ==================================================

            else:

                # Convert to NumPy array

                features = np.asarray(
                    features,
                    dtype=np.float32
                )


                # Flatten

                features = features.flatten()


                # ==================================================
                # DEBUG INFORMATION
                # ==================================================

                print(
                    "Features received:",
                    len(features)
                )

                print(
                    "Model expects:",
                    expected_features
                )


                # ==================================================
                # FEATURE COUNT CHECK
                # ==================================================

                if (
                    expected_features is not None
                    and len(features) != expected_features
                ):

                    st.error(
                        f"❌ Feature mismatch!"
                    )

                    st.write(
                        f"Model expects: "
                        f"{expected_features} features"
                    )

                    st.write(
                        f"Feature extractor returned: "
                        f"{len(features)} features"
                    )

                    st.info(
                        "The model and feature_extraction.py "
                        "were trained using different feature sets."
                    )

                    st.stop()


                # ==================================================
                # PREPARE MODEL INPUT
                # ==================================================

                input_data = features.reshape(
                    1,
                    -1
                )


                # ==================================================
                # MODEL PREDICTION
                # ==================================================

                try:

                    prediction = model.predict(
                        input_data
                    )


                    # ==================================================
                    # PREDICTION PROBABILITY
                    # ==================================================

                    if hasattr(
                        model,
                        "predict_proba"
                    ):

                        probability = model.predict_proba(
                            input_data
                        )[0]


                        # ------------------------------------------
                        # Assuming class 1 = stress
                        # ------------------------------------------

                        if len(probability) > 1:

                            stress_probability = (
                                probability[1] * 100
                            )

                        else:

                            stress_probability = (
                                probability[0] * 100
                            )


                    else:

                        stress_probability = (
                            float(prediction[0]) * 100
                        )


                except Exception as e:

                    st.error(
                        f"Model prediction error: {e}"
                    )

                    break


                # ==================================================
                # CLASSIFY STRESS LEVEL
                # ==================================================

                if stress_probability < 35:

                    level = "Low Stress Risk"

                elif stress_probability < 65:

                    level = "Moderate Stress Risk"

                else:

                    level = "High Stress Risk"


                # ==================================================
                # DISPLAY RESULT ON CAMERA
                # ==================================================

                cv2.putText(
                    frame,
                    level,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"Risk: {stress_probability:.1f}%",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )


                # ==================================================
                # DISPLAY RESULT IN STREAMLIT
                # ==================================================

                prediction_placeholder.metric(
                    "🧠 Stress Risk",
                    f"{stress_probability:.1f}%"
                )


                prediction_placeholder.write(
                    f"### Result: {level}"
                )


            # ==================================================
            # CONVERT BGR TO RGB
            # ==================================================

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # ==================================================
            # DISPLAY CAMERA
            # ==================================================

            frame_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )


    finally:

        camera.release()

        cv2.destroyAllWindows()


# ============================================================
# CAMERA NOT RUNNING
# ============================================================

else:

    st.info(
        "Enable '📷 Start Camera' from the sidebar "
        "to begin real-time analysis."
    )
def get_stress_advice(stress_probability):

    if stress_probability < 35:

        level = "Low Stress Risk"

        advice = [
            "Your current facial cues indicate a low stress-risk level.",
            "Continue maintaining a regular sleep schedule.",
            "Keep yourself physically active and hydrated.",
            "Take short breaks during long periods of study or work."
        ]

    elif stress_probability < 65:

        level = "Moderate Stress Risk"

        advice = [
            "Your current facial cues indicate a moderate stress-risk level.",
            "Take a 5–10 minute break and relax.",
            "Try slow, deep breathing for a few minutes.",
            "Reduce unnecessary screen time and distractions.",
            "Make sure you are getting adequate sleep."
        ]

    else:

        level = "High Stress Risk"

        advice = [
            "Your current facial cues indicate a high stress-risk level.",
            "Take a break from your current activity and move to a comfortable environment.",
            "Try slow breathing: inhale gently for 4 seconds and exhale for 6 seconds.",
            "Drink some water and give yourself time to relax.",
            "Consider talking to a trusted friend, family member, teacher, or counselor if you are feeling overwhelmed.",
            "If high stress continues or is affecting your daily life, consider speaking with a qualified mental-health professional."
        ]

    return level, advice
