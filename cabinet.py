import sys
import numpy as np
from OpenGL.GL import *
import glfw

# Shaders simples: processa posições 2D e cores interpoladas sem matrizes
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

out vec3 ourColor;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    ourColor = aColor;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 ourColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(ourColor, 1.0);
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Erro ao compilar shader: {error}")
    return shader

def create_program():
    vs = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fs = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        error = glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Erro ao linkar programa: {error}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog

class MeshBuilder:
    def __init__(self):
        self.vertices = []

    def add_quad(self, p1, p2, p3, p4, color):
        """Adiciona um quadrilátero dividido em 2 triângulos."""
        r, g, b = color
        pts = [p1, p2, p3, p1, p3, p4]
        for x, y in pts:
            self.vertices.extend([x, y, r, g, b])

    def add_circle(self, cx, cy, rx, ry, color, segments=16):
        """Adiciona puxadores circulares/elípticos."""
        r, g, b = color
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        for i in range(segments):
            a1 = angles[i]
            a2 = angles[(i + 1) % segments]
            p1 = (cx, cy)
            p2 = (cx + rx * np.cos(a1), cy + ry * np.sin(a1))
            p3 = (cx + rx * np.cos(a2), cy + ry * np.sin(a2))
            for x, y in [p1, p2, p3]:
                self.vertices.extend([x, y, r, g, b])

    def get_data(self):
        return np.array(self.vertices, dtype=np.float32)

def build_dresser_geometry():
    mb = MeshBuilder()

    # Paleta de cores com contornos escuros
    c_outline    = (0.10, 0.10, 0.10)
    c_top        = (0.94, 0.92, 0.88)
    c_side       = (0.78, 0.74, 0.67)
    c_side_inset = (0.68, 0.64, 0.58)
    c_front      = (0.83, 0.80, 0.73)
    c_drawer     = (0.89, 0.86, 0.80)
    c_knob       = (0.96, 0.82, 0.15)
    c_mirror_frame = (0.80, 0.75, 0.64)
    c_glass      = (0.42, 0.54, 0.56)
    c_wall_cut   = (0.16, 0.25, 0.24)

    # 1. ESPELHO (posicionado na parede com vão até a cômoda)
    mirror_offset_y = 0.32
    mb.add_quad((-0.88, 0.62 + mirror_offset_y), (-0.42, 0.38 + mirror_offset_y), 
                (-0.42, -0.05 + mirror_offset_y), (-0.88, 0.10 + mirror_offset_y), c_outline)
    mb.add_quad((-0.86, 0.59 + mirror_offset_y), (-0.44, 0.36 + mirror_offset_y), 
                (-0.44, -0.03 + mirror_offset_y), (-0.86, 0.12 + mirror_offset_y), c_mirror_frame)
    mb.add_quad((-0.80, 0.52 + mirror_offset_y), (-0.48, 0.32 + mirror_offset_y), 
                (-0.48, 0.05 + mirror_offset_y), (-0.80, 0.19 + mirror_offset_y), c_outline)
    mb.add_quad((-0.79, 0.50 + mirror_offset_y), (-0.49, 0.31 + mirror_offset_y), 
                (-0.49, 0.07 + mirror_offset_y), (-0.79, 0.20 + mirror_offset_y), c_glass)

    # 2. TAMPO E LATERAL (Sem formas duplicadas e sem sombras externas)
    s_bl = (-0.93, -0.85)
    s_br = (-0.44, -0.88)
    s_tr = (-0.44, 0.16)
    s_tl = (-0.93, 0.19)

    # Tampo superior
    t_back_right = (-0.10, -0.01)
    mb.add_quad(s_tl, s_tr, t_back_right, (-0.60, 0.02), c_outline)
    mb.add_quad(s_tl, (-0.44, 0.15), (-0.11, -0.02), (-0.59, 0.03), c_top)

    # Lateral externa (base sólida)
    mb.add_quad(s_tl, s_tr, s_br, s_bl, c_outline)
    mb.add_quad((-0.92, 0.18), (-0.45, 0.15), (-0.45, -0.87), (-0.92, -0.84), c_side)

    # Painel interno da lateral (único, reto e proporcional)
    mb.add_quad((-0.86, 0.11), (-0.52, 0.09), (-0.52, -0.74), (-0.86, -0.72), c_outline)
    mb.add_quad((-0.85, 0.10), (-0.53, 0.08), (-0.53, -0.73), (-0.85, -0.71), c_side_inset)

    # Recorte da base inferior (pé vazado)
    mb.add_quad((-0.80, -0.85), (-0.57, -0.86), (-0.57, -0.92), (-0.80, -0.92), c_wall_cut)
    mb.add_quad((-0.80, -0.85), (-0.57, -0.86), (-0.57, -0.87), (-0.80, -0.86), c_outline)

    # Contorno da base inferior
    c_thick = 0.012
    mb.add_quad((-0.93, -0.85), (-0.44, -0.88), 
                (-0.44, -0.88 - c_thick), (-0.93, -0.85 - c_thick), c_outline)

    # 3. FRENTE E GAVETAS
    f_tl = (-0.44, 0.16)
    f_tr = (-0.10, -0.01)
    f_br = (-0.10, -0.46)
    f_bl = (-0.44, -0.88)

    mb.add_quad(f_tl, f_tr, f_br, f_bl, c_outline)
    mb.add_quad((-0.43, 0.15), (-0.11, -0.02), (-0.11, -0.45), (-0.43, -0.87), c_front)

    col_split_x = -0.26
    front_left_x = -0.43
    front_right_x = -0.11

    y_lines = [
        (0.13, -0.04),
        (-0.16, -0.16),
        (-0.47, -0.28),
        (-0.78, -0.41)
    ]

    def interp_y(x, y_l, y_r):
        t = (x - front_left_x) / (front_right_x - front_left_x)
        return y_l + t * (y_r - y_l)

    for i in range(3):
        yt_l, yt_r = y_lines[i]
        yb_l, yb_r = y_lines[i+1]

        yt_m = interp_y(col_split_x, yt_l, yt_r)
        yb_m = interp_y(col_split_x, yb_l, yb_r)

        # Coluna Esquerda
        gx1_l, gx1_r = front_left_x + 0.01, col_split_x - 0.008
        gy1_tl = yt_l - 0.015
        gy1_tr = yt_m - 0.015
        gy1_br = yb_m + 0.015
        gy1_bl = yb_l + 0.015

        mb.add_quad((gx1_l, gy1_tl), (gx1_r, gy1_tr), (gx1_r, gy1_br), (gx1_l, gy1_bl), c_outline)
        mb.add_quad((gx1_l + 0.005, gy1_tl - 0.005), (gx1_r - 0.005, gy1_tr - 0.005),
                    (gx1_r - 0.005, gy1_br + 0.005), (gx1_l + 0.005, gy1_bl + 0.005), c_drawer)

        for u in [0.28, 0.72]:
            kx = gx1_l + u * (gx1_r - gx1_l)
            ky_top = gy1_tl + u * (gy1_tr - gy1_tl)
            ky_bot = gy1_bl + u * (gy1_br - gy1_bl)
            ky = (ky_top + ky_bot) * 0.5
            rx = 0.010 - 0.004 * (kx - front_left_x) / (front_right_x - front_left_x)
            ry = 0.015 - 0.005 * (kx - front_left_x) / (front_right_x - front_left_x)
            mb.add_circle(kx, ky, rx, ry, c_knob)

        # Coluna Direita
        gx2_l, gx2_r = col_split_x + 0.008, front_right_x - 0.01
        gy2_tl = yt_m - 0.015
        gy2_tr = yt_r - 0.015
        gy2_br = yb_r + 0.015
        gy2_bl = yb_m + 0.015

        mb.add_quad((gx2_l, gy2_tl), (gx2_r, gy2_tr), (gx2_r, gy2_br), (gx2_l, gy2_bl), c_outline)
        mb.add_quad((gx2_l + 0.005, gy2_tl - 0.005), (gx2_r - 0.005, gy2_tr - 0.005),
                    (gx2_r - 0.005, gy2_br + 0.005), (gx2_l + 0.005, gy2_bl + 0.005), c_drawer)

        for u in [0.28, 0.72]:
            kx = gx2_l + u * (gx2_r - gx2_l)
            ky_top = gy2_tl + u * (gy2_tr - gy2_tl)
            ky_bot = gy2_bl + u * (gy2_br - gy2_bl)
            ky = (ky_top + ky_bot) * 0.5
            rx = 0.010 - 0.004 * (kx - front_left_x) / (front_right_x - front_left_x)
            ry = 0.015 - 0.005 * (kx - front_left_x) / (front_right_x - front_left_x)
            mb.add_circle(kx, ky, rx, ry, c_knob)

    return mb.get_data()

def main():
    if not glfw.init():
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

    width, height = 540, 960
    window = glfw.create_window(width, height, "Comoda OpenGL Ajustada", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)

    program = create_program()
    vertex_data = build_dresser_geometry()
    num_vertices = len(vertex_data) // 5

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    stride = 5 * vertex_data.itemsize
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * vertex_data.itemsize))

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # Fundo do corredor
    glClearColor(0.16, 0.25, 0.24, 1.0)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(program)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, num_vertices)

        glfw.swap_buffers(window)

    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteProgram(program)
    glfw.terminate()

if __name__ == "__main__":
    main()