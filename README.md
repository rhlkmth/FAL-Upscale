# AI Image Upscaler

This Streamlit application uses the [FAL AI](https://fal.ai/) API to upscale and enhance images with modern AI techniques.

## Features

- Upload local photos **or** paste direct image URLs for processing.
- Preview the source image along with its dimensions, mode, and file size before upscaling.
- Configure advanced generation settings such as creativity, resemblance, upscale factor, and custom prompts.
- Monitor queue status and progress updates while the FAL job runs.
- Review a JSON request summary and download the resulting high-resolution image when the job completes.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/FAL-Upscale.git
   cd FAL-Upscale
   ```
2. (Optional) Create and activate a Python virtual environment.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Start the Streamlit development server:

```bash
streamlit run app.py
```

Streamlit will provide a local URL (usually `http://localhost:8501`). Open it in your browser to interact with the UI.

## Usage

1. Generate a FAL API key from the [FAL dashboard](https://fal.ai/dashboard) and enter it in the sidebar (the key is stored only for the active session).
2. Choose how to provide the source image: upload a local file or paste a direct link to an online image.
3. Adjust the advanced settings and optional prompt to guide the upscaler.
4. Review the request summary, then click **Upscale Image**.
5. Monitor the progress indicators. When processing finishes, download the upscaled PNG output.

## Environment Variables

The app reads the `FAL_KEY` environment variable automatically when an API key is provided in the sidebar. You can also set `FAL_KEY` manually before launching the app if you prefer not to enter it in the UI.

## License

This project is provided as-is without a specific license. Please adapt and reuse it to suit your needs while adhering to FAL's API usage policies.
