# OLHA EU SOU UM COMENTÁRIO MUITO IMPORTANTE QUE VAI ENSINAR GIT PRA VC

import sys
import math
import ctypes
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# ==========================================
# SHADERS DO CENÁRIO 2D (CORREDOR)
# ==========================================
CORRIDOR_VERTEX_SRC = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
out vec3 FragColor;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    FragColor = aColor;
}
"""

CORRIDOR_FRAGMENT_SRC = """
#version 330 core
in vec3 FragColor;
out vec4 color;

void main() {
    color = vec4(FragColor, 1.0);
}
"""

# ==========================================
# SHADERS DA CRUZ 3D
# ==========================================
CROSS_VERTEX_SRC = """
#version 330 core
layout (location = 0) in vec3 position;
uniform mat4 mat_transformation;

void main() {
    gl_Position = mat_transformation * vec4(position, 1.0);
}
"""

CROSS_FRAGMENT_SRC = """
#version 330 core
uniform vec4 color;
out vec4 FragColor;

void main() {
    FragColor = color;
}
"""

# ==========================================
# SHADERS DO MONSTRO 2D (COM RUÍDO / GIZ)
# ==========================================
MONSTER_VERTEX_SRC = """
#version 330 core
layout (location = 0) in vec2 aPos;
uniform mat4 uModel;

void main() {
    gl_Position = uModel * vec4(aPos, 0.0, 1.0);
}
"""

MONSTER_FRAGMENT_SRC = """
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

# ==========================================
# VARIÁVEIS DE ESTADO E CONTROLE
# ==========================================
# Cruz 3D
t_x, t_y, t_z = -0.86, 0.1, 0.0
ang_x, ang_y, ang_z = -0.15, -0.5854, 0.15
s_factor = 0.22

# Monstro 2D - Posicionado exatamente no fundo da porta
char_pos = [0.0, -0.05]
char_rotation = 0.0
char_scale = [0.12, 0.12]

wireframe_mode = False

def key_event(window, key, scancode, action, mods):
    global t_x, t_y, ang_x, ang_y, ang_z, s_factor, wireframe_mode
    global char_pos, char_rotation, char_scale

    if action == glfw.PRESS:
        if key == glfw.KEY_P:
            wireframe_mode = not wireframe_mode

    if action in (glfw.PRESS, glfw.REPEAT):
        # --- Controles da Cruz 3D ---
        if key == glfw.KEY_LEFT:  t_x -= 0.02
        if key == glfw.KEY_RIGHT: t_x += 0.02
        if key == glfw.KEY_UP:    t_y += 0.02
        if key == glfw.KEY_DOWN:  t_y -= 0.02

        if key == glfw.KEY_Q: ang_x += 0.05
        if key == glfw.KEY_W: ang_x -= 0.05
        if key == glfw.KEY_A: ang_y += 0.05
        if key == glfw.KEY_S: ang_y -= 0.05
        if key == glfw.KEY_E: ang_z += 0.05
        if key == glfw.KEY_R: ang_z -= 0.05

        if key == glfw.KEY_Z: s_factor = min(3.0, s_factor + 0.02)
        if key == glfw.KEY_X: s_factor = max(0.05, s_factor - 0.02)

        # --- Controles do Monstro 2D ---
        if key == glfw.KEY_L: char_pos[0] += 0.01
        if key == glfw.KEY_J: char_pos[0] -= 0.01
        if key == glfw.KEY_I: char_pos[1] += 0.01
        if key == glfw.KEY_K: char_pos[1] -= 0.01

        if key == glfw.KEY_U: char_rotation += 5.0
        if key == glfw.KEY_O: char_rotation -= 5.0

        if key == glfw.KEY_N: 
            char_scale[0] = min(2.0, char_scale[0] + 0.01)
            char_scale[1] = min(2.0, char_scale[1] + 0.01)
        if key == glfw.KEY_M: 
            char_scale[0] = max(0.02, char_scale[0] - 0.01)
            char_scale[1] = max(0.02, char_scale[1] - 0.01)

# ==========================================
# ÁLGEBRA DE MATRIZES 4X4 (NUMPY)
# ==========================================
def compose_transform_2d(tx=0.0, ty=0.0, angle_deg=0.0, sx=1.0, sy=1.0):
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    
    mat = np.array([
        [cos_a * sx, -sin_a * sy, 0.0, tx],
        [sin_a * sx,  cos_a * sy, 0.0, ty],
        [0.0,         0.0,        1.0, 0.0],
        [0.0,         0.0,        0.0, 1.0]
    ], dtype=np.float32)
    return mat

class Mesh:
    def __init__(self, vao, count, draw_mode=GL_TRIANGLES):
        self.vao = vao
        self.count = count
        self.draw_mode = draw_mode

# ==========================================
# GERAÇÃO DE GEOMETRIAS
# ==========================================
def create_corridor_mesh():
    COLOR_CARPET  = [0.60, 0.08, 0.08]
    COLOR_WALL    = [0.90, 0.90, 0.90]
    COLOR_CEILING = [0.80, 0.80, 0.80]
    COLOR_DOOR    = [0.70, 0.70, 0.70]
    COLOR_BORDER  = [0.10, 0.10, 0.10]

    vertices = [
        -1.0, -1.0, *COLOR_CARPET,   1.0, -1.0, *COLOR_CARPET,   0.18, -0.10, *COLOR_CARPET,
        -1.0, -1.0, *COLOR_CARPET,   0.18, -0.10, *COLOR_CARPET, -0.18, -0.10, *COLOR_CARPET,
        -1.0, -1.0, *COLOR_WALL,    -0.18, -0.10, *COLOR_WALL,  -0.18,  0.10, *COLOR_WALL,
        -1.0, -1.0, *COLOR_WALL,    -0.18,  0.10, *COLOR_WALL,  -1.0,  1.0, *COLOR_WALL,
         0.18, -0.10, *COLOR_WALL,   1.0, -1.0, *COLOR_WALL,     1.0,  1.0, *COLOR_WALL,
         0.18, -0.10, *COLOR_WALL,   1.0,  1.0, *COLOR_WALL,     0.18,  0.10, *COLOR_WALL,
        -1.0,  1.0, *COLOR_CEILING,  1.0,  1.0, *COLOR_CEILING,  0.18,  0.10, *COLOR_CEILING,
        -1.0,  1.0, *COLOR_CEILING,  0.18,  0.10, *COLOR_CEILING,-0.18,  0.10, *COLOR_CEILING,
        -0.18, -0.10, *COLOR_DOOR,   0.18, -0.10, *COLOR_DOOR,   0.18,  0.10, *COLOR_DOOR,
        -0.18, -0.10, *COLOR_DOOR,   0.18,  0.10, *COLOR_DOOR,  -0.18,  0.10, *COLOR_DOOR,
        -1.0, -1.00, *COLOR_BORDER, -0.18, -0.10, *COLOR_BORDER,-0.18, -0.11, *COLOR_BORDER,
        -1.0, -1.00, *COLOR_BORDER, -0.18, -0.11, *COLOR_BORDER,-1.0, -1.02, *COLOR_BORDER,
         1.0, -1.00, *COLOR_BORDER,  0.18, -0.10, *COLOR_BORDER, 0.18, -0.11, *COLOR_BORDER,
         1.0, -1.00, *COLOR_BORDER,  0.18, -0.11, *COLOR_BORDER, 1.0, -1.02, *COLOR_BORDER,
    ]
    return np.array(vertices, dtype=np.float32)

def create_cross_mesh():
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
    return np.array(raw_verts, dtype=np.float32)

def create_quad():
    vertices = np.array([-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5], dtype=np.float32)
    vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return Mesh(vao, 6, GL_TRIANGLES)

def create_triangle():
    vertices = np.array([0.0, 1.0, -0.5, 0.0, 0.5, 0.0], dtype=np.float32)
    vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
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
        vertices.extend([0.5 * math.cos(t * math.pi), t])
    for i in range(segments + 1):
        t = -0.5 + (i / segments)
        vertices.extend([-0.5 * math.cos(t * math.pi), t])
    vertices = np.array(vertices, dtype=np.float32)
    vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
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
        vertices.extend([t, 0.5 * math.cos(t * math.pi)])
    for i in range(segments + 1):
        t = 0.5 - (i / segments)
        vertices.extend([t, -0.5 * math.cos(t * math.pi)])
    vertices = np.array(vertices, dtype=np.float32)
    vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
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
        vertices.extend([math.cos(theta) * 0.5, math.sin(theta) * 0.5])
    vertices = np.array(vertices, dtype=np.float32)
    vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return Mesh(vao, segments + 2, GL_TRIANGLE_FAN)

# ==========================================
# RENDERIZAÇÃO DO MONSTRO
# ==========================================
def draw_shape(mesh, model_matrix, color, model_loc, color_loc, noise_loc, apply_noise=True):
    glUniformMatrix4fv(model_loc, 1, GL_TRUE, model_matrix.flatten())
    glUniform4fv(color_loc, 1, color)
    glUniform1i(noise_loc, 1 if apply_noise else 0)
    glBindVertexArray(mesh.vao)
    glDrawArrays(mesh.draw_mode, 0, mesh.count)

def draw_limb(parent_matrix, start, end, thickness, quad_mesh, model_loc, color_loc, noise_loc, color):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx)) - 90.0
    mid = [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5]
    local_m = compose_transform_2d(tx=mid[0], ty=mid[1], angle_deg=angle, sx=thickness, sy=length)
    draw_shape(quad_mesh, parent_matrix @ local_m, color, model_loc, color_loc, noise_loc, apply_noise=True)

def draw_joint(parent_matrix, pos, radius, circle_mesh, model_loc, color_loc, noise_loc, color):
    local_m = compose_transform_2d(tx=pos[0], ty=pos[1], angle_deg=0.0, sx=radius, sy=radius)
    draw_shape(circle_mesh, parent_matrix @ local_m, color, model_loc, color_loc, noise_loc, apply_noise=True)

def draw_ground_fingers(parent_matrix, wrist_pos, dir_x, triangle_mesh, model_loc, color_loc, noise_loc, body_color):
    angles = [-35.0, -18.0, 0.0, 20.0] if dir_x < 0 else [35.0, 18.0, 0.0, -20.0]
    lengths = [0.22, 0.28, 0.32, 0.25]
    widths  = [0.035, 0.038, 0.040, 0.032]
    for ang, length, width in zip(angles, lengths, widths):
        local_m = compose_transform_2d(tx=wrist_pos[0], ty=wrist_pos[1], angle_deg=(180.0 + ang), sx=width, sy=length)
        draw_shape(triangle_mesh, parent_matrix @ local_m, body_color, model_loc, color_loc, noise_loc, apply_noise=True)

def draw_character(parent_matrix, quad_mesh, triangle_mesh, eye_mesh, circle_mesh, pupil_mesh, model_loc, color_loc, noise_loc):
    black_body = np.array([0.04, 0.04, 0.04, 1.0], dtype=np.float32)
    eye_white  = np.array([0.95, 0.95, 0.95, 1.0], dtype=np.float32)

    # 1. Tronco e Bacia
    draw_limb(parent_matrix, [-0.16, 0.42], [0.14, 0.46], 0.065, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    chest_m = compose_transform_2d(tx=-0.01, ty=0.30, angle_deg=-5.0, sx=0.16, sy=0.24)
    draw_shape(quad_mesh, parent_matrix @ chest_m, black_body, model_loc, color_loc, noise_loc, apply_noise=True)

    draw_limb(parent_matrix, [-0.01, 0.20], [0.05, -0.16], 0.11, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    pelvis_m = compose_transform_2d(tx=0.05, ty=-0.16, angle_deg=10.0, sx=0.15, sy=0.13)
    draw_shape(quad_mesh, parent_matrix @ pelvis_m, black_body, model_loc, color_loc, noise_loc, apply_noise=True)

    # 2. Pescoço
    draw_limb(parent_matrix, [-0.01, 0.44], [-0.03, 0.72], 0.045, quad_mesh, model_loc, color_loc, noise_loc, black_body)

    # 3. Braços Longos
    l_shoulder, l_elbow, l_wrist = [-0.16, 0.42], [-0.25, -0.05], [-0.28, -0.72]
    draw_limb(parent_matrix, l_shoulder, l_elbow, 0.046, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, l_elbow, 0.055, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, l_elbow, l_wrist, 0.038, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, l_wrist, 0.046, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_ground_fingers(parent_matrix, l_wrist, -1.0, triangle_mesh, model_loc, color_loc, noise_loc, black_body)

    r_shoulder, r_elbow, r_wrist = [0.14, 0.46], [0.24, -0.02], [0.26, -0.74]
    draw_limb(parent_matrix, r_shoulder, r_elbow, 0.046, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, r_elbow, 0.055, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, r_elbow, r_wrist, 0.038, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, r_wrist, 0.046, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_ground_fingers(parent_matrix, r_wrist, 1.0, triangle_mesh, model_loc, color_loc, noise_loc, black_body)

    # 4. Pernas
    l_hip, l_knee, l_ankle = [0.01, -0.20], [-0.04, -0.55], [-0.02, -0.88]
    draw_limb(parent_matrix, l_hip, l_knee, 0.048, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, l_knee, 0.054, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, l_knee, l_ankle, 0.038, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, l_ankle, 0.042, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, l_ankle, [-0.08, -0.92], 0.045, quad_mesh, model_loc, color_loc, noise_loc, black_body)

    r_hip, r_knee, r_ankle = [0.09, -0.19], [0.10, -0.53], [0.08, -0.88]
    draw_limb(parent_matrix, r_hip, r_knee, 0.048, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, r_knee, 0.054, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, r_knee, r_ankle, 0.038, quad_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_joint(parent_matrix, r_ankle, 0.042, circle_mesh, model_loc, color_loc, noise_loc, black_body)
    draw_limb(parent_matrix, r_ankle, [0.02, -0.92], 0.045, quad_mesh, model_loc, color_loc, noise_loc, black_body)

    # 5. Cabeça e Espinhos
    head_center = [-0.03, 0.82]
    spike_angles = [0.0, 38.0, -38.0, 180.0, 142.0, 218.0, 90.0, 270.0]
    for angle in spike_angles:
        rad = math.radians(angle)
        pos = [head_center[0] - math.sin(rad) * 0.05, head_center[1] + math.cos(rad) * 0.05]
        spike_m = compose_transform_2d(tx=pos[0], ty=pos[1], angle_deg=angle, sx=0.14, sy=0.30)
        draw_shape(triangle_mesh, parent_matrix @ spike_m, black_body, model_loc, color_loc, noise_loc, apply_noise=True)

    head_base_m = compose_transform_2d(tx=head_center[0], ty=head_center[1], angle_deg=0.0, sx=0.28, sy=0.17)
    draw_shape(circle_mesh, parent_matrix @ head_base_m, black_body, model_loc, color_loc, noise_loc, apply_noise=True)

    eye_m = compose_transform_2d(tx=head_center[0], ty=head_center[1], angle_deg=0.0, sx=0.25, sy=0.12)
    draw_shape(eye_mesh, parent_matrix @ eye_m, eye_white, model_loc, color_loc, noise_loc, apply_noise=False)

    pupil_m = compose_transform_2d(tx=head_center[0], ty=head_center[1], angle_deg=0.0, sx=0.14, sy=0.12)
    draw_shape(pupil_mesh, parent_matrix @ pupil_m, black_body, model_loc, color_loc, noise_loc, apply_noise=True)

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    if not glfw.init():
        sys.exit()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(500, 900, "Corredor + Cruz 3D + Monstro 2D", None, None)
    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_event)

    corridor_shader = compileProgram(
        compileShader(CORRIDOR_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(CORRIDOR_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    cross_shader = compileProgram(
        compileShader(CROSS_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(CROSS_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    monster_shader = compileProgram(
        compileShader(MONSTER_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(MONSTER_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    corridor_verts = create_corridor_mesh()
    corridor_VAO, corridor_VBO = glGenVertexArrays(1), glGenBuffers(1)
    glBindVertexArray(corridor_VAO)
    glBindBuffer(GL_ARRAY_BUFFER, corridor_VBO)
    glBufferData(GL_ARRAY_BUFFER, corridor_verts.nbytes, corridor_verts, GL_STATIC_DRAW)
    stride = 5 * corridor_verts.itemsize
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * corridor_verts.itemsize))
    glEnableVertexAttribArray(1)

    cross_verts = create_cross_mesh()
    cross_VAO, cross_VBO = glGenVertexArrays(1), glGenBuffers(1)
    glBindVertexArray(cross_VAO)
    glBindBuffer(GL_ARRAY_BUFFER, cross_VBO)
    glBufferData(GL_ARRAY_BUFFER, cross_verts.nbytes, cross_verts, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * cross_verts.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    quad_mesh     = create_quad()
    triangle_mesh = create_triangle()
    eye_mesh      = create_eye_shape(40)
    circle_mesh   = create_circle(64)
    pupil_mesh    = create_pupil_shape(40)

    loc_cross_color = glGetUniformLocation(cross_shader, "color")
    loc_cross_trans = glGetUniformLocation(cross_shader, "mat_transformation")

    m_model_loc = glGetUniformLocation(monster_shader, "uModel")
    m_color_loc = glGetUniformLocation(monster_shader, "uColor")
    m_noise_loc = glGetUniformLocation(monster_shader, "uApplyNoise")

    cores_faces = [
        (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, 1.0), (0.5, 0.5, 0.5, 1.0), (0.5, 0.0, 0.0, 1.0),
        (0.8, 0.2, 0.5, 1.0), (0.2, 0.8, 0.8, 1.0), (0.6, 0.3, 0.8, 1.0),
        (1.0, 0.5, 0.0, 1.0), (0.2, 0.5, 0.2, 1.0), (0.3, 0.3, 0.7, 1.0),
        (0.9, 0.4, 0.1, 1.0), (0.1, 0.7, 0.5, 1.0), (0.7, 0.1, 0.7, 1.0),
        (0.4, 0.6, 0.2, 1.0)
    ]

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # -------------------------------------------------------------
        # 1. RENDERIZA CORREDOR 2D
        # -------------------------------------------------------------
        glDisable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glUseProgram(corridor_shader)
        glBindVertexArray(corridor_VAO)
        glDrawArrays(GL_TRIANGLES, 0, len(corridor_verts) // 5)

        # -------------------------------------------------------------
        # 2. RENDERIZA CRUZ 3D
        # -------------------------------------------------------------
        glEnable(GL_DEPTH_TEST)
        glUseProgram(cross_shader)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe_mode else GL_FILL)

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

        mat_composta_cross = mat_translacao @ (mat_rz @ mat_ry @ mat_rx) @ mat_escala
        glUniformMatrix4fv(loc_cross_trans, 1, GL_TRUE, mat_composta_cross.flatten())

        glBindVertexArray(cross_VAO)
        for i in range(16):
            r, g, b, a = cores_faces[i]
            glUniform4f(loc_cross_color, r, g, b, a)
            glDrawArrays(GL_TRIANGLE_STRIP, i * 4, 4)

        # -------------------------------------------------------------
        # 3. RENDERIZA MONSTRO 2D HIERÁRQUICO
        # -------------------------------------------------------------
        # Desativa depth test para garantir visibilidade do olho 2D
        glDisable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glUseProgram(monster_shader)
        
        global_monster_matrix = compose_transform_2d(
            tx=char_pos[0], ty=char_pos[1],
            angle_deg=char_rotation,
            sx=char_scale[0], sy=char_scale[1]
        )

        draw_character(
            global_monster_matrix,
            quad_mesh, triangle_mesh, eye_mesh, circle_mesh, pupil_mesh,
            m_model_loc, m_color_loc, m_noise_loc
        )

        glfw.swap_buffers(window)

    glDeleteVertexArrays(1, [corridor_VAO, cross_VAO])
    glDeleteBuffers(1, [corridor_VBO, cross_VBO])
    glDeleteProgram(corridor_shader)
    glDeleteProgram(cross_shader)
    glDeleteProgram(monster_shader)
    glfw.terminate()

if __name__ == "__main__":
    main()