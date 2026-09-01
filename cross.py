import ctypes
import math
import glfw
import numpy as np
import OpenGL.GL.shaders
from OpenGL.GL import *

MATRIZCORES_FACES = np.array([
    [0.5, 0.07, 0.01, 1.00], [0.0, 0.0, 0.0, 1.00], [0.5, 0.07, 0.01, 1.00],
    [0.0, 0.00, 0.00, 1.00], [0.5, 0.07, 0.01, 1.00], [0.5, 0.07, 0.01, 1.00],
    [0.5, 0.07, 0.01, 1.00], [0.0, 0.0, 0.0, 1.00], [0.5, 0.07, 0.01, 1.00],
    [0.5, 0.07, 0.01, 1.00], [0.5, 0.07, 0.01, 1.00], [0.5, 0.07, 0.01, 1.00],
    [0.0, 0.0, 0.0, 1.00], [0.5, 0.07, 0.01, 1.00], [0.5, 0.07, 0.01, 1.00],
    [0.5, 0.07, 0.01, 1.00]
], dtype=np.float32)


def create_cross_mesh(cores_faces=None):
    if cores_faces is None:
        cores_faces = MATRIZCORES_FACES

    raw_verts = [
        (-0.1, -0.6, +0.1), (+0.1, -0.6, +0.1), (-0.1, +0.6, +0.1), (+0.1, +0.6, +0.1),
        (+0.1, -0.6, +0.1), (+0.1, -0.6, -0.1), (+0.1, +0.6, +0.1), (+0.1, +0.6, -0.1),
        (+0.1, -0.6, -0.1), (-0.1, -0.6, -0.1), (+0.1, +0.6, -0.1), (-0.1, +0.6, -0.1),
        (-0.1, -0.6, -0.1), (-0.1, -0.6, +0.1), (-0.1, +0.6, -0.1), (-0.1, +0.6, +0.1),
        (-0.1, -0.6, -0.1), (+0.1, -0.6, -0.1), (-0.1, -0.6, +0.1), (+0.1, -0.6, +0.1),
        (-0.1, +0.6, +0.1), (+0.1, +0.6, +0.1), (-0.1, +0.6, -0.1), (+0.1, +0.6, -0.1),
        (-0.4, +0.1, +0.1), (-0.1, +0.1, +0.1), (-0.4, +0.3, +0.1), (-0.1, +0.3, +0.1),
        (-0.4, +0.1, -0.1), (-0.4, +0.1, +0.1), (-0.4, +0.3, -0.1), (-0.4, +0.3, +0.1),
        (-0.1, +0.1, -0.1), (-0.4, +0.1, -0.1), (-0.1, +0.3, -0.1), (-0.4, +0.3, -0.1),
        (-0.4, +0.1, -0.1), (-0.1, +0.1, -0.1), (-0.4, +0.1, +0.1), (-0.1, +0.1, +0.1),
        (-0.4, +0.3, +0.1), (-0.1, +0.3, +0.1), (-0.4, +0.3, -0.1), (-0.1, +0.3, -0.1),
        (+0.1, +0.1, +0.1), (+0.4, +0.1, +0.1), (+0.1, +0.3, +0.1), (+0.4, +0.3, +0.1),
        (+0.4, +0.1, +0.1), (+0.4, +0.1, -0.1), (+0.4, +0.3, +0.1), (+0.4, +0.3, -0.1),
        (+0.4, +0.1, -0.1), (+0.1, +0.1, -0.1), (+0.4, +0.3, -0.1), (+0.1, +0.3, -0.1),
        (+0.1, +0.1, -0.1), (+0.4, +0.1, -0.1), (+0.1, +0.1, +0.1), (+0.4, +0.1, +0.1),
        (+0.1, +0.3, +0.1), (+0.4, +0.3, +0.1), (+0.1, +0.3, -0.1), (+0.4, +0.3, -0.1),
    ]

    interleaved_data = []
    for i in range(16):
        cor = cores_faces[i]
        for j in range(4):
            v_idx = i * 4 + j
            x, y, z = raw_verts[v_idx]
            interleaved_data.extend([x, y, z, cor[0], cor[1], cor[2], cor[3]])

    return np.array(interleaved_data, dtype=np.float32)


def run_cross_demo():
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(700, 700, "Cruz 3D - Modo Wireframe (Tecla P)", None, None)

    if window is None:
        print("Failed to create GLFW window")
        glfw.terminate()
        return

    glfw.make_context_current(window)

    vertex_code = """
            attribute vec3 position;
            uniform mat4 mat_transformation;
            void main(){
                gl_Position = mat_transformation * vec4(position, 1.0);
            }
            """

    fragment_code = """
            uniform vec4 color;
            void main(){
                gl_FragColor = color;
            }
            """

    program = glCreateProgram()
    vertex = glCreateShader(GL_VERTEX_SHADER)
    fragment = glCreateShader(GL_FRAGMENT_SHADER)

    glShaderSource(vertex, vertex_code)
    glShaderSource(fragment, fragment_code)

    glCompileShader(vertex)
    if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(vertex).decode())

    glCompileShader(fragment)
    if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(fragment).decode())

    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError("Linking error")

    glUseProgram(program)

    vertices = np.zeros(64, [("position", np.float32, 3)])
    vertices["position"] = [
        (-0.1, -0.6, +0.1), (+0.1, -0.6, +0.1), (-0.1, +0.6, +0.1), (+0.1, +0.6, +0.1),
        (+0.1, -0.6, +0.1), (+0.1, -0.6, -0.1), (+0.1, +0.6, +0.1), (+0.1, +0.6, -0.1),
        (+0.1, -0.6, -0.1), (-0.1, -0.6, -0.1), (+0.1, +0.6, -0.1), (-0.1, +0.6, -0.1),
        (-0.1, -0.6, -0.1), (-0.1, -0.6, +0.1), (-0.1, +0.6, -0.1), (-0.1, +0.6, +0.1),
        (-0.1, -0.6, -0.1), (+0.1, -0.6, -0.1), (-0.1, -0.6, +0.1), (+0.1, -0.6, +0.1),
        (-0.1, +0.6, +0.1), (+0.1, +0.6, +0.1), (-0.1, +0.6, -0.1), (+0.1, +0.6, -0.1),
        (-0.4, +0.1, +0.1), (-0.1, +0.1, +0.1), (-0.4, +0.3, +0.1), (-0.1, +0.3, +0.1),
        (-0.4, +0.1, -0.1), (-0.4, +0.1, +0.1), (-0.4, +0.3, -0.1), (-0.4, +0.3, +0.1),
        (-0.1, +0.1, -0.1), (-0.4, +0.1, -0.1), (-0.1, +0.3, -0.1), (-0.4, +0.3, -0.1),
        (-0.4, +0.1, -0.1), (-0.1, +0.1, -0.1), (-0.4, +0.1, +0.1), (-0.1, +0.1, +0.1),
        (-0.4, +0.3, +0.1), (-0.1, +0.3, +0.1), (-0.4, +0.3, -0.1), (-0.1, +0.3, -0.1),
        (+0.1, +0.1, +0.1), (+0.4, +0.1, +0.1), (+0.1, +0.3, +0.1), (+0.4, +0.3, +0.1),
        (+0.4, +0.1, +0.1), (+0.4, +0.1, -0.1), (+0.4, +0.3, +0.1), (+0.4, +0.3, -0.1),
        (+0.4, +0.1, -0.1), (+0.1, +0.1, -0.1), (+0.4, +0.3, -0.1), (+0.1, +0.3, -0.1),
        (+0.1, +0.1, -0.1), (+0.4, +0.1, -0.1), (+0.1, +0.1, +0.1), (+0.4, +0.1, +0.1),
        (+0.1, +0.3, +0.1), (+0.4, +0.3, +0.1), (+0.1, +0.3, -0.1), (+0.4, +0.3, -0.1),
    ]

    buffer_VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer_VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

    stride = vertices.strides[0]
    offset = ctypes.c_void_p(0)
    loc = glGetAttribLocation(program, "position")
    glEnableVertexAttribArray(loc)
    glVertexAttribPointer(loc, 3, GL_FLOAT, False, stride, offset)

    loc_color = glGetUniformLocation(program, "color")
    loc_transformation = glGetUniformLocation(program, "mat_transformation")

    t_x, t_y, t_z = 0.0, 0.0, 0.0
    ang_x, ang_y, ang_z = 0.0, 0.0, 0.0
    s_factor = 1.0
    wireframe_mode = False

    def key_event(window, key, scancode, action, mods):
        nonlocal t_x, t_y, ang_x, ang_y, ang_z, s_factor, wireframe_mode
        if action == glfw.PRESS and key == glfw.KEY_P:
            wireframe_mode = not wireframe_mode
        if action in (glfw.PRESS, glfw.REPEAT):
            if key == glfw.KEY_LEFT: t_x -= 0.02
            if key == glfw.KEY_RIGHT: t_x += 0.02
            if key == glfw.KEY_UP: t_y += 0.02
            if key == glfw.KEY_DOWN: t_y -= 0.02
            if key == glfw.KEY_Q: ang_x += 0.05
            if key == glfw.KEY_W: ang_x -= 0.05
            if key == glfw.KEY_A: ang_y += 0.05
            if key == glfw.KEY_S: ang_y -= 0.05
            if key == glfw.KEY_E: ang_z += 0.05
            if key == glfw.KEY_R: ang_z -= 0.05
            if key == glfw.KEY_Z: s_factor = min(3.0, s_factor + 0.05)
            if key == glfw.KEY_X: s_factor = max(0.1, s_factor - 0.05)

    glfw.set_key_callback(window, key_event)
    glfw.show_window(window)
    glEnable(GL_DEPTH_TEST)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.9, 0.9, 0.9, 1.0)

        if wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        mat_escala = np.array([
            [s_factor, 0.0, 0.0, 0.0],
            [0.0, s_factor, 0.0, 0.0],
            [0.0, 0.0, s_factor, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        cos_x, sin_x = math.cos(ang_x), math.sin(ang_x)
        mat_rx = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_x, -sin_x, 0.0],
            [0.0, sin_x, cos_x, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        cos_y, sin_y = math.cos(ang_y), math.sin(ang_y)
        mat_ry = np.array([
            [cos_y, 0.0, sin_y, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_y, 0.0, cos_y, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        cos_z, sin_z = math.cos(ang_z), math.sin(ang_z)
        mat_rz = np.array([
            [cos_z, -sin_z, 0.0, 0.0],
            [sin_z, cos_z, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        mat_translacao = np.array([
            [1.0, 0.0, 0.0, t_x],
            [0.0, 1.0, 0.0, t_y],
            [0.0, 0.0, 1.0, t_z],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        mat_composta = mat_translacao @ (mat_rz @ mat_ry @ mat_rx) @ mat_escala
        glUniformMatrix4fv(loc_transformation, 1, GL_TRUE, mat_composta.flatten())

        for i in range(16):
            r, g, b, a = MATRIZCORES_FACES[i]
            glUniform4f(loc_color, r, g, b, a)
            glDrawArrays(GL_TRIANGLE_STRIP, i * 4, 4)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    run_cross_demo()