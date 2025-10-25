import gradio as gr
from utils.unet_utils import predict_unet_from_file
from utils.geo_utils import predict_geo
import os
import cv2
import numpy as np

def run_app(file, threshold=0.65):
    if file is None:
        return None, "Please upload a file first."
    
    try:
        if file.name.endswith((".dcm", ".npy", ".png")):
            # Enhanced prediction with crop reconstruction
            overlay, meta = predict_unet_from_file(
                file.name, 
                models_dir="./models", 
                threshold=threshold
            )
            
            # Enhanced metadata display
            enhanced_meta = f"""### Model Results 📊
**Threshold Used:** {threshold:.2f}
**Volume Fraction:** {meta.get('vol_frac_percent', 0):.2f}%
**Model File:** {meta.get('model_file', 'N/A')}
**Has Crop Metadata:** {'✅ Yes' if meta.get('has_metadata', False) else '❌ No (estimated)'}
**Patient ID:** {meta.get('patient_id', 'Unknown')}
**Slice Index:** {meta.get('slice_index', 'Unknown')}
**Original Shape:** {meta.get('orig_shape', 'N/A')}
**Status:** Processing Complete

### Recommendations 💡
- If segmentation too large → increase threshold
- If missing parts → decrease threshold  
- Optimal range: 0.5 - 0.8
- {'Crop metadata found - accurate positioning!' if meta.get('has_metadata') else 'No metadata - using estimated region'}
"""
            
            return overlay, enhanced_meta
            
        elif file.name.endswith(".csv"):
            result = predict_geo(file.name, models_dir="./models")
            return None, str(result)
        else:
            return None, "Unsupported file format. Please upload .dcm, .npy, .png, or .csv files."
            
    except Exception as e:
        return None, f"Error: {str(e)}"

# Gradio interface
with gr.Blocks(title="Pancreas AI Assistant with Accurate Positioning", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🥞 Pancreas AI Assistant with Accurate Positioning")
    gr.Markdown("Upload a medical image (.dcm, .npy, .png) or genomic data (.csv) for AI analysis.")
    
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.File(
                label="Upload Medical File",
                file_types=[".dcm", ".npy", ".png", ".csv"]
            )
            
            threshold_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.65,
                step=0.05,
                label="Segmentation Threshold",
                info="Lower = more sensitive, Higher = more specific"
            )
            
            gr.Markdown("### Instructions:")
            gr.Markdown("- **.dcm/.npy/.png**: Medical images for segmentation")
            gr.Markdown("- **.csv**: Genomic data for survival prediction") 
            gr.Markdown("- **Threshold**: Adjust for optimal segmentation")
            
        with gr.Column(scale=2):
            out_img = gr.Image(
                label="Segmentation Overlay",
                type="pil"
            )
            
            out_text = gr.Markdown(
                label="Analysis Results",
                value="Upload a file to see results..."
            )
    
    # Interactive updates
    inp.change(
        run_app, 
        inputs=[inp, threshold_slider], 
        outputs=[out_img, out_text]
    )
    
    threshold_slider.change(
        run_app,
        inputs=[inp, threshold_slider], 
        outputs=[out_img, out_text]
    )
    
    # Example files section
    with gr.Row():
        gr.Markdown("### Example Usage:")
        gr.Markdown("1. Upload a DICOM/NPY file from TCIA pancreas dataset")
        gr.Markdown("2. Adjust threshold if needed (0.65 is optimal for most cases)")
        gr.Markdown("3. Check if crop metadata was found for accurate positioning")
        gr.Markdown("4. Download results if needed")

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./utils", exist_ok=True)
    
    demo.launch(
        inbrowser=True,
        share=False,
        debug=False,
        show_error=True,
        quiet=False,
        server_name="0.0.0.0",
        server_port=7860
    )