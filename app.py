import streamlit as st
import fal_client
import os
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import base64

# Config and settings
st.set_page_config(page_title="Image Upscaler", layout="wide")


def format_bytes(num_bytes):
    """Return a human friendly representation of a byte value."""
    step = 1024.0
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < step or unit == "TB":
            if unit == "B":
                return f"{int(num_bytes)} {unit}"
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= step


@st.cache_data(show_spinner=False)
def fetch_image_from_url(url: str) -> bytes:
    """Download an image from a URL and return the image bytes."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError("Unable to download the image. Please verify the URL.") from exc

    content_bytes = response.content
    try:
        Image.open(BytesIO(content_bytes))
    except Exception as exc:
        raise ValueError("The provided link does not contain a valid image.") from exc

    return content_bytes


def describe_image(metadata):
    """Build a short description for an image."""
    width, height = metadata["dimensions"]
    size_label = format_bytes(metadata["size"])
    mode = metadata["mode"]
    return f"Dimensions: {width}×{height}px · Mode: {mode} · File Size: {size_label}"

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
    st.markdown(
        "Need an API key? [Generate one from the FAL dashboard](https://fal.ai/dashboard)."
    )
    api_key = st.text_input(
        "Enter FAL API Key:",
        type="password",
        placeholder="sk-...",
        help="Your key is stored only for this session and never shared.",
    )
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

# Track the currently selected image
input_image = None
input_metadata = {}

# Input column
with col1:
    st.header("Input Image")
    source_option = st.radio(
        "Select image source",
        options=["Upload", "Image URL"],
        horizontal=True,
    )

    if source_option == "Upload":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["png", "jpg", "jpeg", "webp"],
            help="Supported formats: PNG, JPG, JPEG and WEBP.",
        )
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image.load()
                input_metadata = {
                    "dimensions": image.size,
                    "mode": image.mode,
                    "source": uploaded_file.name,
                    "size": uploaded_file.size,
                }
                input_image = image.copy()
                image.close()
                st.image(input_image, caption="Input Preview", use_column_width=True)
                st.caption(describe_image(input_metadata))
            except UnidentifiedImageError:
                st.error("The selected file is not a supported image format.")
            finally:
                uploaded_file.seek(0)
    else:
        image_url = st.text_input(
            "Paste an image URL",
            placeholder="https://example.com/photo.png",
            help="Provide a direct link to a PNG, JPG, JPEG or WEBP image.",
        )
        if image_url:
            try:
                image_bytes = fetch_image_from_url(image_url)
                image = Image.open(BytesIO(image_bytes))
                image.load()
                input_metadata = {
                    "dimensions": image.size,
                    "mode": image.mode,
                    "source": image_url,
                    "size": len(image_bytes),
                }
                input_image = image.copy()
                image.close()
                st.image(input_image, caption="Input Preview", use_column_width=True)
                st.caption(describe_image(input_metadata))
            except ValueError as error:
                st.error(str(error))

    if input_image is None:
        st.info("Upload an image or provide a URL to get started.")

# Output column
with col2:
    st.header("Upscaled Result")
    if input_metadata:
        st.caption(f"Selected image • {describe_image(input_metadata)}")
    st.caption(
        f"Creativity: {creativity:.2f} · Resemblance: {resemblance:.2f} · Upscale ×{upscale_factor:.1f}"
    )

    request_summary = {
        "prompt": prompt,
        "creativity": creativity,
        "resemblance": resemblance,
        "upscale_factor": upscale_factor,
    }

    if input_metadata:
        request_summary["input_image"] = {
            "source": input_metadata.get("source"),
            "dimensions": list(input_metadata.get("dimensions", ())),
            "mode": input_metadata.get("mode"),
            "size_bytes": input_metadata.get("size"),
        }

    with st.expander("Request summary", expanded=False):
        st.json(request_summary)

    button_disabled = input_image is None or not api_key
    button_help = (
        "Start by providing an image and API key."
        if button_disabled
        else "Send the upscale request to FAL AI."
    )

    if st.button(
        "Upscale Image",
        disabled=button_disabled,
        help=button_help,
        use_container_width=True,
    ):
        if button_disabled:
            if not api_key:
                st.warning("Please enter your FAL API key in the sidebar.")
            if input_image is None:
                st.warning("Upload an image or provide a valid URL to continue.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            queued_state = getattr(fal_client, "Queued", None)

            def on_queue_update(update):
                if queued_state and isinstance(update, queued_state):
                    position = getattr(update, "position", None)
                    if position is not None:
                        status_text.info(f"Queued... {position} job(s) ahead")
                    else:
                        status_text.info("Queued... waiting for a worker")
                    progress_bar.progress(10)
                elif isinstance(update, fal_client.InProgress):
                    messages = [
                        log.get("message") for log in update.logs if isinstance(log, dict)
                    ]
                    if messages:
                        status_text.info(f"Processing: {messages[-1]}")
                    progress_bar.progress(55)

            try:
                with st.spinner("Upscaling image..."):
                    image_uri = image_to_data_uri(input_image)

                    result = fal_client.subscribe(
                        "fal-ai/clarity-upscaler",
                        arguments={
                            "image_url": image_uri,
                            "prompt": prompt,
                            "creativity": creativity,
                            "resemblance": resemblance,
                            "upscale_factor": upscale_factor,
                            "enable_safety_checker": False,
                            "guidance_scale": 4,
                            "num_inference_steps": 18,
                        },
                        with_logs=True,
                        on_queue_update=on_queue_update,
                    )

                progress_bar.progress(85)

                image_info = None
                if isinstance(result, dict):
                    if "image" in result:
                        image_info = result["image"]
                    elif isinstance(result.get("images"), list) and result["images"]:
                        image_info = result["images"][0]

                if image_info and "url" in image_info:
                    response = requests.get(image_info["url"], timeout=60)
                    response.raise_for_status()
                    output_image = Image.open(BytesIO(response.content))
                    st.image(output_image, caption="Upscaled Result", use_column_width=True)
                    st.success(f"New Size: {output_image.size}")
                    status_text.success("Upscaling complete!")

                    buf = BytesIO()
                    output_image.save(buf, format="PNG")
                    st.download_button(
                        label="Download Upscaled Image",
                        data=buf.getvalue(),
                        file_name="upscaled_image.png",
                        mime="image/png",
                    )
                    progress_bar.progress(100)
                else:
                    status_text.warning("No result received from the service.")
                    st.warning("The upscaling service did not return an image. Please try again.")

            except Exception as exc:
                status_text.error("Upscaling failed.")
                st.error("An error occurred while processing the image.")
                st.error(str(exc))
            finally:
                progress_bar.empty()

# Footer
st.markdown("---")
st.markdown("### 📝 Instructions")
st.markdown("""
1. Enter your FAL API key in the sidebar.
2. Upload an image **or** paste an image URL.
3. Adjust advanced settings to guide the upscaler.
4. Review the request summary and click **Upscale Image**.
5. Download your enhanced result once processing finishes.
""")
