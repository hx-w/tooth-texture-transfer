# -*- coding: utf-8 -*-

import os
import gradio as gr

import transfer_tex as tt

class GradioUI:
    def __init__(self):
        theme = gr.themes.Soft(
            primary_hue="slate",
        ).set(
            checkbox_background_color_dark='*neutral_900'
        )
        self.inst = gr.Blocks(theme=theme)
        self.__build()

    def __build(self):
        self._mesh_uploaded = 0

        with self.inst:
            gr.Markdown(
                '# 上前切牙纹理（细节）迁移\n'
                '输入： **带纹理的牙齿 $M_A$** 和 **光滑牙齿 $M_B$**\n<br/>'
                '输出：**$M_A$的牙齿移位贴图** 和 **带纹理的牙齿 $M_B\prime$**\n<br/><br/>'
                '注意：输入的牙齿网格不需要对齐，也不需要是封闭网格。但需要保证都是11牙，且网格的开闭性质相同。<br/>'
                '**需要确保输入的网格是二维流形，且至多有一个边界**'
            )
            with gr.Row(equal_height=False):
                # Input part
                with gr.Column():
                    gr.Markdown('### 输入区')
                    tex_mesh = gr.Model3D(label='纹理牙齿网格',)
                    smt_mesh = gr.Model3D(label='平滑牙齿网格')
                    tex_resol = gr.Slider(50.0, 300.0, step=50.0, value=100.0, label='纹理精度（nxn）')
                    btn_commit = gr.Button('开始迁移', interactive=False)
                
                with gr.Column():
                    gr.Markdown('### 输出区')
                    tex_image = gr.Image(label='移位贴图')
                    trans_mesh = gr.Model3D(label='纹理迁移网格')
            
            ## methods
            def on_upload_mesh(tex_mesh_path: str, smt_mesh_path: str):
                def _check(m_path: str) -> bool:
                    if not isinstance(m_path, str):
                        return False
                    if os.path.splitext(m_path)[-1] != '.obj':
                        gr.Warning('非OBJ格式的3D模型暂不支持前端渲染，不影响结果')
                    return os.path.isfile(m_path)
                
                return { btn_commit: gr.update(interactive=_check(tex_mesh_path) and _check(smt_mesh_path))}
            
            def on_submit_click(tex_mesh_path: str, smt_mesh_path: str, tex_resol: int, progress=gr.Progress(track_tqdm=True)):
                tex_imagepath, trans_meshpath = tt.procceed(tex_mesh_path, smt_mesh_path, tex_resol, progress)
                
                return {
                    tex_image: tex_imagepath,
                    trans_mesh: trans_meshpath
                }
            
            tex_mesh.change(on_upload_mesh, inputs=[tex_mesh, smt_mesh], outputs=[btn_commit])
            smt_mesh.change(on_upload_mesh, inputs=[tex_mesh, smt_mesh], outputs=[btn_commit])
            btn_commit.click(on_submit_click, inputs=[tex_mesh, smt_mesh, tex_resol], outputs=[tex_image, trans_mesh])
              
    def launch_public(self, host='127.0.0.1', port=8001):
        self.inst.queue().launch(share=False, server_name=host, server_port=port)



if __name__ == '__main__':
    GradioUI().launch_public()
