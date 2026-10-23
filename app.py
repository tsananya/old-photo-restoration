import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import os
import shutil
from restore_pipeline import inpaint_damage, restore_faces, correct_color, upscale_image, enhance_clarity



st.set_page_config(page_title="Old Photo Restoration", layout="wide")
st.title("🖼️ Old Photo Restoration")
st.write("Upload an old photo, mark any damaged areas (scratches, stains, tears), and restore it.")

WORK_DIR = "app_temp"
os.makedirs(WORK_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    full_image = image
    max_side = int(os.environ.get("MAX_SIDE", 1500))
    if max(image.size) > max_side:
        s = max_side / max(image.size)
        image = image.resize((int(image.width * s), int(image.height * s)), Image.LANCZOS)
    mode = st.radio("Photo type", ["Damaged photo (full restoration)", "Clean photo (light enhance)"], horizontal=True)
    if mode.startswith("Clean"):
        sharp = st.slider("Sharpness", 0.0, 1.5, 0.8, 0.1, key="sharp_clean")
        contrast = st.slider("Local contrast", 0.0, 1.0, 0.4, 0.1, key="contrast_clean")
        clean_scale = st.selectbox("Upscale (1 = none)", [1, 1.5, 2], index=0, key="scale_clean")
        if st.button("Enhance", key="enhance_btn"):
            with st.spinner("Enhancing..."):
                clean_img = full_image
                limit = int(os.environ.get("CLEAN_MAX_SIDE", 6000)) / clean_scale
                if max(clean_img.size) > limit:
                    k = limit / max(clean_img.size)
                    clean_img = clean_img.resize((int(clean_img.width * k), int(clean_img.height * k)), Image.LANCZOS)
                clean_dir = f"{WORK_DIR}/clean"
                if os.path.exists(clean_dir):
                    shutil.rmtree(clean_dir)
                os.makedirs(f"{clean_dir}/in")
                in_path = f"{clean_dir}/in/clean_in.jpg"
                clean_img.save(in_path, quality=95)
                src = in_path
                if clean_scale > 1:
                    upscale_image(
                        input_folder=f"../{clean_dir}/in",
                        output_folder=f"../{clean_dir}/up",
                        outscale=clean_scale
                    )
                    src = f"{clean_dir}/up/" + os.listdir(f"{clean_dir}/up")[0]
                out_path = f"{clean_dir}/clean_out.jpg"
                enhance_clarity(src, out_path, sharpen=sharp, contrast=contrast)
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Original**")
                st.image(clean_img, use_container_width=True)
            with c2:
                st.write("**Enhanced**")
                st.image(Image.open(out_path), use_container_width=True)
            with open(out_path, "rb") as f:
                st.download_button("Download", f, file_name="enhanced.jpg", mime="image/jpeg")
        st.stop()

    # Resize for display/masking (keep aspect ratio, cap width for canvas usability)
    display_width = 600
    scale = display_width / image.width
    display_height = int(image.height * scale)
    display_image = image.resize((display_width, display_height))

    st.write("### Step 1: Mark damaged areas")
    st.write("Draw over scratches, stains, or tears you want repaired. Leave undamaged areas untouched.")

    brush_size = st.slider("Brush size", 5, 50, 20)

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=brush_size,
        stroke_color="rgba(255, 255, 255, 1)",
        background_image=display_image,
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True,
    )

    fidelity = st.slider("Face fidelity (higher = closer to original face)", 0.3, 1.0, 0.7, 0.05)
    color_strength = st.slider("Color correction strength (0 = off)", 0.0, 1.0, 0.0, 0.1)
    outscale = st.selectbox("Upscale factor (1 = none, fastest)", [1, 1.5, 2, 4], index=0)
    clarity = st.slider("Clarity / sharpness (0 = off)", 0.0, 1.5, 0.8, 0.1)

    if st.button("Restore Photo"):
        if canvas_result.image_data is None:
            st.error("Please mark at least one damaged area first.")
        else:
            with st.spinner("Restoring... this may take a few minutes."):
                # Clear previous run
                if os.path.exists(WORK_DIR):
                    shutil.rmtree(WORK_DIR)
                os.makedirs(WORK_DIR)

                # Save original at full resolution
                original_path = f"{WORK_DIR}/original.jpg"
                image.save(original_path)

                # Build mask from canvas, resized back to original resolution
                mask_canvas = canvas_result.image_data[:, :, 3]  # alpha channel = drawn strokes
                mask_resized = cv2.resize(mask_canvas, (image.width, image.height))
                mask_binary = (mask_resized > 0).astype(np.uint8) * 255
                mask_path = f"{WORK_DIR}/mask.png"
                cv2.imwrite(mask_path, mask_binary)

                # Step 1: Inpaint
                step1_output = f"{WORK_DIR}/step1_inpainted.jpg"
                inpaint_damage(original_path, mask_path, step1_output)

                # Step 2: Face restoration
                step2_input = f"{WORK_DIR}/step2_input"
                os.makedirs(step2_input, exist_ok=True)
                shutil.copy(step1_output, step2_input)
                restore_faces(
                    input_folder=f"../{step2_input}",
                    output_folder=f"../{WORK_DIR}/step2_output",
                    fidelity=fidelity
                )
                step2_result = f"{WORK_DIR}/step2_output/final_results/step1_inpainted.png"

                # Step 3: Color correction
                step3_output = f"{WORK_DIR}/step3_colorcorrected.jpg"
                correct_color(step2_result, step3_output, mask_path=mask_path, strength=color_strength)

                # Step 4: Upscale
                                # Step 4: Upscale (optional)
                if outscale == 1:
                    final_path = step3_output
                else:
                    step4_input = f"{WORK_DIR}/step4_input"
                    os.makedirs(step4_input, exist_ok=True)
                    shutil.copy(step3_output, step4_input)
                    upscale_image(
                        input_folder=f"../{step4_input}",
                        output_folder=f"../{WORK_DIR}/step4_final",
                        outscale=outscale
                    )
                    final_filename = os.listdir(f"{WORK_DIR}/step4_final")[0]
                    final_path = f"{WORK_DIR}/step4_final/{final_filename}"
                                # Step 5: Clarity (sharpen brightness only, colours untouched)
                if clarity > 0:
                    clear_path = f"{WORK_DIR}/final_clear.jpg"
                    enhance_clarity(final_path, clear_path, sharpen=clarity, contrast=0.0)
                    final_path = clear_path
            st.success("Restoration complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original**")
                st.image(image, use_container_width=True)
            with col2:
                st.write("**Restored**")
                result_image = Image.open(final_path)
                st.image(result_image, use_container_width=True)

            with open(final_path, "rb") as f:
                st.download_button("Download Restored Photo", f, file_name="restored.jpg", mime="image/jpeg")