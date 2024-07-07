import copy
from modules import scripts_postprocessing
from PIL import Image
import gradio as gr
from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from lama_cleaner_masked_content.options import getLamaUpscaler
from lama_cleaner_masked_content.inpaint import lamaInpaint, limitSizeByMinDimension


if hasattr(scripts_postprocessing.ScriptPostprocessing, 'process_firstpass'):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None



def get_current_image(image):
    if image is None:
        return
    maxResolutionOnDetection = 1024
    image = decode_base64_to_image(image)
    image = limitSizeByMinDimension(image, maxResolutionOnDetection)
    image = 'data:image/png;base64,' + encode_pil_to_base64(image).decode()
    return gr.Image.update(image)



class ScriptPostprocessing(scripts_postprocessing.ScriptPostprocessing):
    name = 'Lama cleaner'
    order = 110000

    def ui(self):
        with (
            InputAccordion(False, label=self.name, elem_id='lama_cleaner_extras') if InputAccordion
            else gr.Accordion(self.name, open=False, elem_id='lama_cleaner_extras')
            as enable
        ):
            with gr.Row():
                if not InputAccordion:
                    enable = gr.Checkbox(False, label='Enable')
            with gr.Row():
                create_canvas = gr.Button('Create canvas')
                mask_source = gr.CheckboxGroup(['Draw mask', 'Upload mask'], value=['Draw mask'], label="Canvas mask source")
                mask_brush_color = gr.ColorPicker('#84FF9A', label='Brush color', info='visual only, use when brush color is hard to see')
            with gr.Row():
                input_mask = gr.Image(
                    label="Mask",
                    show_label=False,
                    elem_id="lama_cleaner_extras_mask",
                    source="upload",
                    interactive=True,
                    type="pil",
                    tool="sketch",
                    image_mode="RGBA",
                    brush_color='#84FF9A'
                )

            with gr.Row():
                padding = gr.Slider(label="Padding", minimum=-1, maximum=512, value=90, step=1, info='-1 for no padding')
                invert = gr.Checkbox(label="Invert mask", value=False)

            def update_mask_brush_color(color):
                return gr.Image.update(brush_color=color)

            mask_brush_color.change(
                fn=update_mask_brush_color,
                inputs=[mask_brush_color],
                outputs=[input_mask]
            )

            dummy_component = gr.Label(visible=False)
            create_canvas.click(
                fn=get_current_image,
                _js='getCurrentExtraSourceImg_lama_cleaner',
                inputs=[dummy_component],
                outputs=[input_mask],
                postprocess=False,
            )

        controls = {
            'enable': enable,
            'mask_source': mask_source,
            'input_mask': input_mask,
            'padding': padding,
            'invert': invert,
        }
        return controls

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if not args['enable']:
            return
        padding = None
        if args['padding'] != -1:
            padding = args['padding']
        invert = args['invert']

        mask = None

        if args['input_mask']:
            if 'Upload mask' in args['mask_source']:
                mask = args['input_mask']['image'].convert('L').resize(pp.image.size)
            if 'Draw mask' in args['mask_source']:
                mask = Image.new('L', pp.image.size, 0) if mask is None else mask
                draw_mask = args['input_mask']['mask'].convert('L').resize(pp.image.size)
                mask.paste(draw_mask, draw_mask)

        if not mask: return
        pp.image = lamaInpaint(pp.image, mask, invert, getLamaUpscaler(), padding)
        pp.info[self.name] = f'padding={padding}, invert={invert}'

