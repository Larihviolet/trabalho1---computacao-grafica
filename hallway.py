import sys
import ctypes
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# Shaders simples sem necessidade de uniform de projeção
VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

out vec3 FragColor;

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    FragColor = aColor;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 FragColor;
out vec4 color;

void main()
{
    color = vec4(FragColor, 1.0);
}
"""

def create_corridor_mesh():
    COLOR_CARPET  = [0.60, 0.08, 0.08]
    COLOR_WALL    = [0.90, 0.90, 0.90]
    COLOR_CEILING = [0.80, 0.80, 0.80]
    COLOR_DOOR    = [0.70, 0.70, 0.70]
    COLOR_BORDER  = [0.10, 0.10, 0.10]

    # Para a janela 500x900 (aspect ratio 5/9), o fundo precisa de y_half = x_half * (5/9) 
    # para ser visualmente um quadrado perfeito (ex: x em [-0.18, 0.18] e y em [-0.10, 0.10])
    vertices = [
        # --- CHÃO (Vermelho) ---
        -1.0, -1.0, *COLOR_CARPET,
         1.0, -1.0, *COLOR_CARPET,
         0.18, -0.10, *COLOR_CARPET,
        
        -1.0, -1.0, *COLOR_CARPET,
         0.18, -0.10, *COLOR_CARPET,
        -0.18, -0.10, *COLOR_CARPET,

        # --- PAREDE ESQUERDA ---
        -1.0, -1.0, *COLOR_WALL,
        -0.18, -0.10, *COLOR_WALL,
        -0.18,  0.10, *COLOR_WALL,

        -1.0, -1.0, *COLOR_WALL,
        -0.18,  0.10, *COLOR_WALL,
        -1.0,  1.0, *COLOR_WALL,

        # --- PAREDE DIREITA ---
         0.18, -0.10, *COLOR_WALL,
         1.0, -1.0, *COLOR_WALL,
         1.0,  1.0, *COLOR_WALL,

         0.18, -0.10, *COLOR_WALL,
         1.0,  1.0, *COLOR_WALL,
         0.18,  0.10, *COLOR_WALL,

        # --- TETO ---
        -1.0,  1.0, *COLOR_CEILING,
         1.0,  1.0, *COLOR_CEILING,
         0.18,  0.10, *COLOR_CEILING,

        -1.0,  1.0, *COLOR_CEILING,
         0.18,  0.10, *COLOR_CEILING,
        -0.18,  0.10, *COLOR_CEILING,

        # --- PAREDE DO FUNDO (QUADRADO VISUAL) ---
        -0.18, -0.10, *COLOR_DOOR,
         0.18, -0.10, *COLOR_DOOR,
         0.18,  0.10, *COLOR_DOOR,

        -0.18, -0.10, *COLOR_DOOR,
         0.18,  0.10, *COLOR_DOOR,
        -0.18,  0.10, *COLOR_DOOR,

        # --- RODAPÉ ESQUERDO ---
        -1.0, -1.00, *COLOR_BORDER,
        -0.18, -0.10, *COLOR_BORDER,
        -0.18, -0.11, *COLOR_BORDER,

        -1.0, -1.00, *COLOR_BORDER,
        -0.18, -0.11, *COLOR_BORDER,
        -1.0, -1.02, *COLOR_BORDER,

        # --- RODAPÉ DIREITO ---
         1.0, -1.00, *COLOR_BORDER,
         0.18, -0.10, *COLOR_BORDER,
         0.18, -0.11, *COLOR_BORDER,

         1.0, -1.00, *COLOR_BORDER,
         0.18, -0.11, *COLOR_BORDER,
         1.0, -1.02, *COLOR_BORDER,
    ]

    return np.array(vertices, dtype=np.float32)

def main():
    if not glfw.init():
        sys.exit()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(500, 900, "Corredor 2D - Fundo Quadrado", None, None)
    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    shader = compileProgram(
        compileShader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER)
    )

    vertices = create_corridor_mesh()

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    stride = 5 * vertices.itemsize
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader)

        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 5)

        glfw.swap_buffers(window)

    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader)
    glfw.terminate()

if __name__ == "__main__":
    main()