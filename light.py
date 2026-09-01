import sys
import math
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# --- SHADERS ---
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 uTransform;
out vec3 fragColor;

void main() {
    gl_Position = uTransform * vec4(aPos, 1.0);
    fragColor = aColor;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(fragColor, 1.0);
}
"""

# --- MATRIZES DE TRANSFORMAÇÃO PRIMÁRIAS ---
def matrix_scale(sx, sy, sz):
    return np.array([
        [sx,  0,  0, 0],
        [ 0, sy,  0, 0],
        [ 0,  0, sz, 0],
        [ 0,  0,  0, 1]
    ], dtype=np.float32)

def matrix_rotate_x(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [1,  0,  0, 0],
        [0,  c, -s, 0],
        [0,  s,  c, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)

def matrix_translate(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0,  1]
    ], dtype=np.float32)

# --- GEOMETRIA DA LUMINÁRIA (COM ESPESSURA E PROFUNDIDADE) ---
def generate_lamp_geometry(segments=64):
    vertices = []
    
    color_light = [1.0, 0.98, 0.88]  # Painel emissivo (frente)
    color_rim   = [1.0, 1.0, 1.0]  # Borda lateral externa
    color_back  = [1.0, 1.0, 0.4]  # Fundo/base (sombra sutil)

    # 1. Painel Frontal (Disco Iluminado em Z = +0.2)
    vertices.extend([0.0, 0.0, 0.2, *color_light])
    for i in range(segments + 1):
        angle = (2.0 * math.pi * i) / segments
        x = math.cos(angle)
        y = math.sin(angle)
        vertices.extend([x, y, 0.2, *color_light])

    # 2. Borda Lateral Extensa (Cilindro de Z = +0.2 até Z = -0.3 para manter o corpo)
    rim_start = segments + 2
    for i in range(segments + 1):
        angle = (2.0 * math.pi * i) / segments
        x = math.cos(angle)
        y = math.sin(angle)
        
        vertices.extend([x, y, 0.2, *color_rim])
        vertices.extend([x, y, -0.3, *color_rim])

    # 3. Tampa Traseira (Garante o fechamento do volume no fundo)
    back_start = rim_start + (segments + 1) * 2
    vertices.extend([0.0, 0.0, -0.3, *color_back])
    for i in range(segments + 1):
        angle = (2.0 * math.pi * i) / segments
        x = math.cos(angle)
        y = math.sin(angle)
        vertices.extend([x, y, -0.3, *color_back])

    return np.array(vertices, dtype=np.float32), segments, rim_start, back_start

def main():
    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 800, "Luminaria 3D - Profundidade Preservada", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    
    # Habilita teste de profundidade para renderização 3D correta
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    vertices, segments, rim_start, back_start = generate_lamp_geometry()

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    stride = 6 * vertices.itemsize
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    # --- MONTAGEM DA MATRIZ DE TRANSFORMAÇÃO COMPOSTA ---
    # Escala: dá o tamanho total e define a profundidade da borda no eixo Z
    S = matrix_scale(0.70, 0.70, 0.40)
    
    # Rotação: inclina a luminária (~65º) para expor tanto o painel quanto a lateral
    R_x = matrix_rotate_x(math.radians(-65))
    
    # Translação: reposiciona no topo da tela
    T = matrix_translate(0.0, 0.15, 0.0)

    # Matriz Composta (T * R * S)
    M_composite = T @ R_x @ S

    transform_loc = glGetUniformLocation(shader, "uTransform")
    glClearColor(0.88, 0.86, 0.82, 1.0)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)
        glUniformMatrix4fv(transform_loc, 1, GL_TRUE, M_composite)
        glBindVertexArray(VAO)

        # Desenha a tampa traseira
        glDrawArrays(GL_TRIANGLE_FAN, back_start, segments + 2)

        # Desenha a borda lateral (cilindro)
        glDrawArrays(GL_TRIANGLE_STRIP, rim_start, (segments + 1) * 2)

        # Desenha o painel luminoso frontal por último
        glDrawArrays(GL_TRIANGLE_FAN, 0, segments + 2)

        glfw.swap_buffers(window)

    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glfw.terminate()

if __name__ == "__main__":
    main()