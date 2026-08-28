import sys
import subprocess

# 1. Instalação automática das dependências caso não existam
REQUIRED_PACKAGES = ["PyOpenGL", "PyOpenGL_accelerate", "glfw", "PyGLM", "numpy"]

def install_packages():
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.lower() if package != "PyGLM" else "glm")
        except ImportError:
            print(f"Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

import glfw
from OpenGL.GL import *
import glm
import numpy as np
import math
import ctypes

SCR_WIDTH = 800
SCR_HEIGHT = 1000

# Estado global para alternar a exibição da malha (wireframe) e transformações interativas
wireframe_mode = False

# Variáveis para a Transformação Composta Global do Personagem
char_pos = glm.vec2(0.0, 0.0)
char_rotation = 0.0
char_scale = glm.vec2(1.0, 1.0)

def key_callback(window, key, scancode, action, mods):
    global wireframe_mode, char_pos, char_rotation, char_scale
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_P and action == glfw.PRESS:
            wireframe_mode = not wireframe_mode
            if wireframe_mode:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # Translação primária do personagem
        elif key == glfw.KEY_RIGHT: char_pos.x += 0.04
        elif key == glfw.KEY_LEFT:  char_pos.x -= 0.04
        elif key == glfw.KEY_UP:    char_pos.y += 0.04
        elif key == glfw.KEY_DOWN:  char_pos.y -= 0.04
        
        # Rotação primária do personagem
        elif key == glfw.KEY_A:     char_rotation += 5.0
        elif key == glfw.KEY_D:     char_rotation -= 5.0
        
        # Escala primária do personagem
        elif key == glfw.KEY_W:     char_scale += glm.vec2(0.05, 0.05)
        elif key == glfw.KEY_S:     char_scale = glm.max(char_scale - glm.vec2(0.05, 0.05), glm.vec2(0.1, 0.1))

# --- FUNÇÃO DE TRANSFORMAÇÃO COMPOSTA PRIMÁRIA (M = T * R * S) ---
def compose_transform(translation=glm.vec2(0.0, 0.0), angle_degrees=0.0, scale=glm.vec2(1.0, 1.0)):
    # 1. Matriz de Translação Primária (T)
    T = glm.translate(glm.mat4(1.0), glm.vec3(translation.x, translation.y, 0.0))
    
    # 2. Matriz de Rotação Primária (R)
    R = glm.rotate(glm.mat4(1.0), math.radians(angle_degrees), glm.vec3(0.0, 0.0, 1.0))
    
    # 3. Matriz de Escala Primária (S)
    S = glm.scale(glm.mat4(1.0), glm.vec3(scale.x, scale.y, 1.0))
    
    # Matriz Composta Final
    return T * R * S

# Shaders com Ruído Procedural (Giz / Carvão)
VERTEX_SHADER_SOURCE = """
#version 330 core
layout (location = 0) in vec2 aPos;

uniform mat4 uProjection;
uniform mat4 uModel;

void main() {
    gl_Position = uProjection * uModel * vec4(aPos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330 core
out vec4 FragColor;

uniform vec4 uColor;
uniform int uApplyNoise;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main() {
    if (uApplyNoise == 1) {
        float noise = random(gl_FragCoord.xy * 0.4);
        float factor = mix(0.70, 1.30, noise);
        vec3 finalColor = uColor.rgb * factor;
        FragColor = vec4(finalColor, uColor.a);
    } else {
        FragColor = uColor;
    }
}
"""

class Mesh:
    def __init__(self, vao, count, draw_mode=GL_TRIANGLES):
        self.vao = vao
        self.count = count
        self.draw_mode = draw_mode

def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode('utf-8'))
    return shader

def create_shader_program():
    vs = compile_shader(GL_VERTEX_SHADER, VERTEX_SHADER_SOURCE)
    fs = compile_shader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program).decode('utf-8'))
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

def create_quad():
    vertices = np.array([
        -0.5, -0.5,
         0.5, -0.5,
         0.5,  0.5,
        -0.5, -0.5,
         0.5,  0.5,
        -0.5,  0.5
    ], dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return Mesh(vao, 6, GL_TRIANGLES)

def create_triangle():
    vertices = np.array([
         0.0,  1.0,
        -0.5,  0.0,
         0.5,  0.0
    ], dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return Mesh(vao, 3, GL_TRIANGLES)

def create_pupil_shape(segments=40):
    vertices = [0.0, 0.0]
    for i in range(segments + 1):
        t = 0.5 - (i / segments)
        x = 0.5 * math.cos(t * math.pi)
        vertices.extend([x, t])
        
    for i in range(segments + 1):
        t = -0.5 + (i / segments)
        x = -0.5 * math.cos(t * math.pi)
        vertices.extend([x, t])

    vertices = np.array(vertices, dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return Mesh(vao, (segments + 1) * 2 + 1, GL_TRIANGLE_FAN)

def create_eye_shape(segments=40):
    vertices = [0.0, 0.0]
    for i in range(segments + 1):
        t = -0.5 + (i / segments)
        y = 0.5 * math.cos(t * math.pi)
        vertices.extend([t, y])
        
    for i in range(segments + 1):
        t = 0.5 - (i / segments)
        y = -0.5 * math.cos(t * math.pi)
        vertices.extend([t, y])

    vertices = np.array(vertices, dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return Mesh(vao, (segments + 1) * 2 + 1, GL_TRIANGLE_FAN)

def create_circle(segments=48):
    vertices = [0.0, 0.0]
    for i in range(segments + 1):
        theta = 2.0 * math.pi * float(i) / float(segments)
        vertices.append(math.cos(theta) * 0.5)
        vertices.append(math.sin(theta) * 0.5)

    vertices = np.array(vertices, dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return Mesh(vao, segments + 2, GL_TRIANGLE_FAN)

def draw_shape(mesh, model_matrix, color, model_loc, color_loc, noise_loc, apply_noise=True):
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matrix))
    glUniform4fv(color_loc, 1, glm.value_ptr(color))
    glUniform1i(noise_loc, 1 if apply_noise else 0)
    glBindVertexArray(mesh.vao)
    glDrawArrays(mesh.draw_mode, 0, mesh.count)

def draw_limb(parent_matrix, start, end, thickness, quad_mesh, model_loc, color_loc, noise_loc, color):
    diff = end - start
    length = glm.length(diff)
    angle = math.degrees(atan2_val := math.atan2(diff.y, diff.x)) - 90.0
    mid = (start + end) * 0.5

    # Matriz composta local: T * R * S
    local_m = compose_transform(translation=mid, angle_degrees=angle, scale=glm.vec2(thickness, length))
    draw_shape(quad_mesh, parent_matrix * local_m, color, model_loc, color_loc, noise_loc, apply_noise=True)

def draw_joint(parent_matrix, pos, radius, circle_mesh, model_loc, color_loc, noise_loc, color):
    local_m = compose_transform(translation=pos, angle_degrees=0.0, scale=glm.vec2(radius, radius))
    draw_shape(circle_mesh, parent_matrix * local_m, color, model_loc, color_loc, noise_loc, apply_noise=True)

def draw_ground_fingers(parent_matrix, wrist_pos, dir_x, triangle_mesh, model_loc, color_loc, noise_loc, body_color):
    angles = [-35.0, -18.0, 0.0, 20.0] if dir_x < 0 else [35.0, 18.0, 0.0, -20.0]
    lengths = [0.22, 0.28, 0.32, 0.25]
    widths  = [0.035, 0.038, 0.040, 0.032]
    
    for ang, length, width in zip(angles, lengths, widths):
        local_m = compose_transform(translation=wrist_pos, angle_degrees=(180.0 + ang), scale=glm.vec2(width, length))
        draw_shape(triangle_mesh, parent_matrix * local_m, body_color, model_loc, color_loc, noise_loc, apply_noise=True)

def draw_character(parent_matrix, quad_mesh, triangle_mesh, eye_mesh, circle_mesh, pupil_mesh, model_loc, color_loc, noise_loc):
    black_body = glm.vec4(0.04, 0.04, 0.04, 1.0)
    eye_white  = glm.vec4(0.95, 0.95, 0.95, 1.0)

    # --- 1. TRONCO E BACIA ---
    draw_limb(parent_matrix, glm.vec2(-0.16, 0.42), glm.vec2(0.14, 0.46), 0.065, quad_mesh, model_loc, color_loc, noise_loc, black_body)

    chest_m = compose_transform(translation=glm.vec2(-0.01, 0.30), angle_degrees=-5.0, scale=glm.vec2(0.16, 0.24))
    draw_shape(quad_mesh, parent_matrix * chest_m, black_body, model_loc, color_loc, noise_loc, apply_noise=True)

    draw_limb(parent_matrix, glm.vec2(-0.01, 0.20), glm.vec2(0.05, -0.16), 0.11, quad_mesh, model_loc, color_loc, noise_loc, black_body)

    pelvis_m = compose_transform(translation=glm.vec2(0.05, -0.16), angle_degrees=10.0, scale=glm.vec2(0.15, 0.13))
    draw_shape(quad_mesh, parent_matrix * pelvis_m, black_body, model_loc, color_loc, noise_loc, apply_noise=True)

    # --- 2. PESCOÇO ---
    draw_limb(parent_matrix, glm.vec2(-0.01, 0.44), glm.vec2(-0.03, 0.72), 0.045, quad_mesh, model_loc, color_loc, noise_loc, black_body)

    # --- 3. BRAÇOS LONGOS ---
    l_shoulder = glm.vec2(-0.16, 0.42)
    l_elbow    = glm.vec2(-0.25, -0.05)
    l_wrist    = glm.vec2(-0.28, -0.72)
    draw_limb(parent_matrix, l_shoulder, l_elbow, 0.046, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, l_elbow, 0.055, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, l_elbow, l_wrist, 0.038, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, l_wrist, 0.046, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_ground_fingers(parent_matrix, l_wrist, -1.0, triangle_mesh, model_loc, color_loc, noise_loc, black_body)

    r_shoulder = glm.vec2(0.14, 0.46)
    r_elbow    = glm.vec2(0.24, -0.02)
    r_wrist    = glm.vec2(0.26, -0.74)
    draw_limb(parent_matrix, r_shoulder, r_elbow, 0.046, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, r_elbow, 0.055, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, r_elbow, r_wrist, 0.038, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, r_wrist, 0.046, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_ground_fingers(parent_matrix, r_wrist, 1.0, triangle_mesh, model_loc, color_loc, noise_loc, black_body)

    # --- 4. PERNAS ---
    l_hip   = glm.vec2(0.01, -0.20)
    l_knee  = glm.vec2(-0.04, -0.55)
    l_ankle = glm.vec2(-0.02, -0.88)
    draw_limb(parent_matrix, l_hip, l_knee, 0.048, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, l_knee, 0.054, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, l_knee, l_ankle, 0.038, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, l_ankle, 0.042, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, l_ankle, glm.vec2(-0.08, -0.92), 0.045, quad_mesh, model_loc, color_loc, noise_loc, black_body)

    r_hip   = glm.vec2(0.09, -0.19)
    r_knee  = glm.vec2(0.10, -0.53)
    r_ankle = glm.vec2(0.08, -0.88)
    draw_limb(parent_matrix, r_hip, r_knee, 0.048, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, r_knee, 0.054, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, r_knee, r_ankle, 0.038, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, r_ankle, 0.042, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, r_ankle, glm.vec2(0.02, -0.92), 0.045, quad_mesh, model_loc, color_loc, noise_loc, black_body)

    # --- 5. CABEÇA ---
    head_center = glm.vec2(-0.03, 0.82)
    spike_angles = [0.0, 38.0, -38.0, 180.0, 142.0, 218.0, 90.0, 270.0]
    spike_length = 0.30
    spike_width  = 0.14
    spike_offset = 0.05

    for angle in spike_angles:
        rad = math.radians(angle)
        pos = head_center + glm.vec2(-math.sin(rad), math.cos(rad)) * spike_offset
        spike_m = compose_transform(translation=pos, angle_degrees=angle, scale=glm.vec2(spike_width, spike_length))
        draw_shape(triangle_mesh, parent_matrix * spike_m, black_body, model_loc, color_loc, noise_loc, apply_noise=True)

    head_base_m = compose_transform(translation=head_center, angle_degrees=0.0, scale=glm.vec2(0.28, 0.17))
    draw_shape(circle_mesh, parent_matrix * head_base_m, black_body, model_loc, color_loc, noise_loc, apply_noise=True)

    eye_m = compose_transform(translation=head_center, angle_degrees=0.0, scale=glm.vec2(0.25, 0.12))
    draw_shape(eye_mesh, parent_matrix * eye_m, eye_white, model_loc, color_loc, noise_loc, apply_noise=False)

    pupil_m = compose_transform(translation=head_center, angle_degrees=0.0, scale=glm.vec2(0.14, 0.12))
    draw_shape(pupil_mesh, parent_matrix * pupil_m, black_body, model_loc, color_loc, noise_loc, apply_noise=True)

def main():
    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(SCR_WIDTH, SCR_HEIGHT, "Character 2D - Matrizes Compostas (P: Wireframe | Setas: Move | A/D: Roda | W/S: Escala)", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)

    shader_program = create_shader_program()
    model_loc = glGetUniformLocation(shader_program, "uModel")
    color_loc = glGetUniformLocation(shader_program, "uColor")
    proj_loc  = glGetUniformLocation(shader_program, "uProjection")
    noise_loc = glGetUniformLocation(shader_program, "uApplyNoise")

    quad_mesh     = create_quad()
    triangle_mesh = create_triangle()
    eye_mesh      = create_eye_shape(40)
    circle_mesh   = create_circle(64)
    pupil_mesh    = create_pupil_shape(40)

    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT)

    while not glfw.window_should_close(window):
        glClearColor(0.96, 0.96, 0.96, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)

        aspect = SCR_WIDTH / SCR_HEIGHT
        view_height = 1.15
        projection = glm.ortho(-aspect * view_height, aspect * view_height, -view_height, view_height)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))

        # 1. Matriz Composta Global do Personagem (T * R * S)
        global_character_matrix = compose_transform(
            translation=char_pos,
            angle_degrees=char_rotation,
            scale=char_scale
        )

        # 2. Desenho de todas as partes hierarquicamente compostas
        draw_character(global_character_matrix, quad_mesh, triangle_mesh, eye_mesh, circle_mesh, pupil_mesh, model_loc, color_loc, noise_loc)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()