import streamlit as st
import fal_client
import os
from PIL import Image
import requests
from io import BytesIO
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Config and settings
st.set_page_config(page_title="Image Upscaler", layout="wide")

def image_to_data_uri(img):
    """Convert a PIL Image to a data URI."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Streamlit UI layout
st.title("🖼️ AI Image Upscaler")

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter FAL API Key:", type="password")
    if api_key:
        os.environ["FAL_KEY"] = api_key
    
    # Advanced settings collapsible
    with st.expander("Advanced Settings"):
        creativity = st.slider("Creativity", 0.0, 1.0, 0.35)
        resemblance = st.slider("Resemblance", 0.0, 1.0, 0.6)
        upscale_factor = st.slider("Upscale Factor", 1.0, 4.0, 2.0)
        prompt = st.text_area("Custom Prompt", "masterpiece, best quality, highres")

# Main content area - two columns
col1, col2 = st.columns(2)

# Input column
with col1:
    st.header("Input Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display the input image
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Input Image", use_column_width=True)
        st.info(f"Original Size: {input_image.size}")

# Output column
with col2:
    st.header("Upscaled Result")
    
    # Process button
    if st.button("Upscale Image", disabled=not (uploaded_file and api_key)):
        if uploaded_file and api_key:
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process the image
                with st.spinner("Upscaling image..."):
                    # Convert image to data URI
                    image_uri = image_to_data_uri(input_image)
                    
                    # Run upscaling
                    result = fal.run(
                        "fal-ai/clarity-upscaler",
                        input={
                            "image_url": image_uri,
                            "prompt": prompt,
                            "creativity": creativity,
                            "resemblance": resemblance,
                            "upscale_factor": upscale_factor,
                            "enable_safety_checker": False,
                            "guidance_scale": 4,
                            "num_inference_steps": 18,
                        }
                    )
                    
                    logging.info(f"API Response: {result}")
                    
                    # Show result
                    if result and 'image' in result:
                        # Download and display the result
                        response = requests.get(result['image'])
                        output_image = Image.open(BytesIO(response.content))
                        st.image(output_image, caption="Upscaled Result", use_column_width=True)
                        st.success(f"New Size: {output_image.size}")
                        
                        # Add download button
                        buf = BytesIO()
                        output_image.save(buf, format="PNG")
                        st.download_button(
                            label="Download Upscaled Image",
                            data=buf.getvalue(),
                            file_name="upscaled_image.png",
                            mime="image/png"
                        )
                    else:
                        st.error("No image in the result. Please check the API response.")
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
            except Exception as e:
                logging.error(f"Error occurred: {str(e)}", exc_info=True)
                st.error(f"Error occurred: {str(e)}")
                st.error("Please check the logs for more details.")
                progress_bar.empty()
                status_text.empty()
        else:
            st.warning("Please upload an image and enter your API key.")

# Footer
st.markdown("---")
st.markdown("### 📝 Instructions")
st.markdown("""
1. Enter your FAL API key in the sidebar
2. Upload an image you want to upscale
3. Adjust advanced settings if desired
4. Click 'Upscale Image' to process
5. Download the result when ready
""")
