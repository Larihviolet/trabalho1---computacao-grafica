import sys
import math
import ctypes
import subprocess

# ==============================================================================
# MECANISMO DE INSTALAÇÃO AUTOMÁTICA
# ==============================================================================
try:
    import numpy as np
    import glfw
    from OpenGL.GL import *
    from OpenGL.GL.shaders import compileProgram, compileShader
except ImportError as e:
    print(f"Pacote faltando detectado: {e.name}. Instalando dependências...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "glfw",
                "PyOpenGL",
                "PyOpenGL_accelerate",
                "numpy",
            ]
        )
        print("Instalação concluída com sucesso! Reiniciando o programa...\n")
        subprocess.call([sys.executable] + sys.argv)
        sys.exit(0)
    except Exception as install_error:
        print(f"\nErro na instalação automática: {install_error}")
        print("Por favor, instale manualmente: pip install glfw PyOpenGL PyOpenGL_accelerate numpy")
        sys.exit(1)

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
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor; // Atributo de cor do VBO

out vec4 vColor; // Envia a cor para o Fragment Shader

uniform mat4 mat_transformation;

void main() {
    gl_Position = mat_transformation * vec4(aPos, 1.0);
    vColor = aColor;
}
"""

CROSS_FRAGMENT_SRC = """
#version 330 core
in vec4 vColor;
out vec4 FragColor;

void main() {
    FragColor = vColor;
}
"""

# ==========================================
# SHADERS DE OBJETOS 2D HIERÁRQUICOS (MONSTRO E ESQUELETO)
# ==========================================
CHAR_VERTEX_SRC = """
#version 330 core
layout (location = 0) in vec2 aPos;
uniform mat4 uModel;

void main() {
    gl_Position = uModel * vec4(aPos, 0.0, 1.0);
}
"""

CHAR_FRAGMENT_SRC = """
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

# Monstro 2D
char_pos = [0.0, -0.05]
char_rotation = 0.0
char_scale = [0.12, 0.12]

wireframe_mode = False

# --- Esqueleto 2D ---
skel_state = {
    'tx': 0.78,
    'ty': -0.72,
    'rotation': 0.0,
    'scale': 1.0,
}

skel_head_state = {
    'falling': False,
    'x': 0.0,
    'y': 0.65,
    'vx': 0.0,
    'vy': 0.0,
    'rotation': 0.0,
    'hit_ground': False,
    'target_rotation': 0.0,
}

SKEL_GROUND_Y = -0.95
SKEL_GRAVITY = -0.0016
SKEL_LEFT_BOUND = -0.98


def trigger_skel_head_fall():
    if not skel_head_state['falling']:
        skel_head_state['falling'] = True
        rad = math.radians(skel_state['rotation'])
        local_x, local_y = 0.0, 0.65
        scaled_x = local_x * skel_state['scale']
        scaled_y = local_y * skel_state['scale']
        world_x = skel_state['tx'] + (scaled_x * math.cos(rad) - scaled_y * math.sin(rad))
        world_y = skel_state['ty'] + (scaled_x * math.sin(rad) + scaled_y * math.cos(rad))

        skel_head_state['x'] = world_x
        skel_head_state['y'] = world_y
        skel_head_state['vx'] = -0.006
        skel_head_state['vy'] = 0.0
        skel_head_state['rotation'] = 0.0
        skel_head_state['hit_ground'] = False
        skel_head_state['target_rotation'] = 0.0


def reset_skeleton():
    skel_state['tx'] = 0.78
    skel_state['ty'] = -0.72
    skel_state['rotation'] = 0.0
    skel_state['scale'] = 1.0

    skel_head_state['falling'] = False
    skel_head_state['x'] = 0.0
    skel_head_state['y'] = 0.65
    skel_head_state['vx'] = 0.0
    skel_head_state['vy'] = 0.0
    skel_head_state['rotation'] = 0.0
    skel_head_state['hit_ground'] = False
    skel_head_state['target_rotation'] = 0.0


def update_skeleton_physics():
    if not skel_head_state['falling']:
        return

    head_radius = 0.08 * skel_state['scale']

    if not skel_head_state['hit_ground']:
        skel_head_state['vy'] += SKEL_GRAVITY

    skel_head_state['x'] += skel_head_state['vx']
    skel_head_state['y'] += skel_head_state['vy']

    if skel_head_state['y'] - head_radius <= SKEL_GROUND_Y:
        skel_head_state['y'] = SKEL_GROUND_Y + head_radius
        skel_head_state['vy'] = 0.0

        if not skel_head_state['hit_ground']:
            skel_head_state['hit_ground'] = True
            skel_head_state['target_rotation'] = skel_head_state['rotation'] + (360.0 * 3)

    if skel_head_state['hit_ground']:
        if skel_head_state['rotation'] < skel_head_state['target_rotation']:
            remaining = skel_head_state['target_rotation'] - skel_head_state['rotation']
            speed_factor = max(0.05, remaining / (360.0 * 3))

            rot_step = 8.0 * speed_factor
            skel_head_state['rotation'] += rot_step

            dx = math.radians(rot_step) * head_radius
            skel_head_state['x'] -= dx
        else:
            skel_head_state['rotation'] = skel_head_state['target_rotation']
            skel_head_state['vx'] = 0.0

    if skel_head_state['x'] - head_radius <= SKEL_LEFT_BOUND:
        skel_head_state['x'] = SKEL_LEFT_BOUND + head_radius
        skel_head_state['vx'] = 0.0
        skel_head_state['target_rotation'] = skel_head_state['rotation']


def key_event(window, key, scancode, action, mods):
    global t_x, t_y, ang_x, ang_y, ang_z, s_factor, wireframe_mode
    global char_pos, char_rotation, char_scale

    if action == glfw.PRESS:
        if key == glfw.KEY_P:
            wireframe_mode = not wireframe_mode
        if key == glfw.KEY_C:
            trigger_skel_head_fall()
        if key == glfw.KEY_0:
            reset_skeleton()
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

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

        # --- Controles do Esqueleto 2D ---
        if key == glfw.KEY_T: skel_state['ty'] += 0.01
        if key == glfw.KEY_G: skel_state['ty'] -= 0.01
        if key == glfw.KEY_F: skel_state['tx'] -= 0.01
        if key == glfw.KEY_H: skel_state['tx'] += 0.01

        if key == glfw.KEY_Y: skel_state['rotation'] += 2.0
        if key == glfw.KEY_V: skel_state['rotation'] -= 2.0

        if key == glfw.KEY_PERIOD: skel_state['scale'] = min(2.0, skel_state['scale'] + 0.01)
        if key == glfw.KEY_COMMA:  skel_state['scale'] = max(0.05, skel_state['scale'] - 0.01)


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


def _upload_2d_mesh(vertices, draw_mode):
    vertices = np.array(vertices, dtype=np.float32)
    vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return vao, len(vertices) // 2


# ==========================================
# GERAÇÃO DE GEOMETRIAS
# ==========================================
def create_corridor_mesh():
    COLOR_CARPET  = [0.2, 0.2, 0.2]
    COLOR_WALL    = [0.18, 0.31, 0.31]
    COLOR_CEILING = [0.16, 0.29, 0.29]
    COLOR_DOOR    = [0.08, 0.14, 0.14]

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
    ]
    return np.array(vertices, dtype=np.float32)


def create_cross_mesh(cores_faces):
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

    # Intercala posição (x, y, z) com cor (r, g, b, a)
    for i in range(16): # 16 faces
        cor = cores_faces[i]
        for j in range(4): # 4 vértices por face
            v_idx = i * 4 + j
            x, y, z = raw_verts[v_idx]
            interleaved_data.extend([x, y, z, cor[0], cor[1], cor[2], cor[3]])

    return np.array(interleaved_data, dtype=np.float32)

def create_quad():
    vertices = [-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5]
    vao, count = _upload_2d_mesh(vertices, GL_TRIANGLES)
    return Mesh(vao, count, GL_TRIANGLES)


def create_triangle():
    vertices = [0.0, 1.0, -0.5, 0.0, 0.5, 0.0]
    vao, count = _upload_2d_mesh(vertices, GL_TRIANGLES)
    return Mesh(vao, count, GL_TRIANGLES)


def create_pupil_shape(segments=40):
    vertices = [0.0, 0.0]
    for i in range(segments + 1):
        t = 0.5 - (i / segments)
        vertices.extend([0.5 * math.cos(t * math.pi), t])
    for i in range(segments + 1):
        t = -0.5 + (i / segments)
        vertices.extend([-0.5 * math.cos(t * math.pi), t])
    vao, count = _upload_2d_mesh(vertices, GL_TRIANGLE_FAN)
    return Mesh(vao, count, GL_TRIANGLE_FAN)


def create_eye_shape(segments=40):
    vertices = [0.0, 0.0]
    for i in range(segments + 1):
        t = -0.5 + (i / segments)
        vertices.extend([t, 0.5 * math.cos(t * math.pi)])
    for i in range(segments + 1):
        t = 0.5 - (i / segments)
        vertices.extend([t, -0.5 * math.cos(t * math.pi)])
    vao, count = _upload_2d_mesh(vertices, GL_TRIANGLE_FAN)
    return Mesh(vao, count, GL_TRIANGLE_FAN)


def create_circle(segments=48):
    vertices = [0.0, 0.0]
    for i in range(segments + 1):
        theta = 2.0 * math.pi * float(i) / float(segments)
        vertices.extend([math.cos(theta) * 0.5, math.sin(theta) * 0.5])
    vao, count = _upload_2d_mesh(vertices, GL_TRIANGLE_FAN)
    return Mesh(vao, count, GL_TRIANGLE_FAN)


def create_arc_mesh(segments=10):
    vertices = []
    for i in range(segments + 1):
        t = i / segments
        x = -0.5 * math.cos(t * math.pi)
        y = math.sin(t * math.pi)
        vertices.extend([x, y])
    vao, count = _upload_2d_mesh(vertices, GL_LINE_STRIP)
    return Mesh(vao, count, GL_LINE_STRIP)


def create_spine_mesh(segments=10):
    spine_top_x, spine_top_y = 0.02, 0.58
    control_x, control_y = 0.10, 0.30
    pelvis_x, pelvis_y = 0.0, 0.0

    vertices = []
    for i in range(segments + 1):
        t = i / segments
        sx = (1 - t) ** 2 * spine_top_x + 2 * (1 - t) * t * control_x + t ** 2 * pelvis_x
        sy = (1 - t) ** 2 * spine_top_y + 2 * (1 - t) * t * control_y + t ** 2 * pelvis_y
        vertices.extend([sx, sy])
    vao, count = _upload_2d_mesh(vertices, GL_LINE_STRIP)
    return Mesh(vao, count, GL_LINE_STRIP)


# ==========================================
# RENDERIZAÇÃO DE OBJETOS HIERÁRQUICOS
# ==========================================
def draw_shape(mesh, model_matrix, color, model_loc, color_loc, noise_loc, apply_noise=True):
    glUniformMatrix4fv(model_loc, 1, GL_TRUE, model_matrix.flatten())
    glUniform4fv(color_loc, 1, color)
    glUniform1i(noise_loc, 1 if (apply_noise and not wireframe_mode) else 0)
    glBindVertexArray(mesh.vao)
    glDrawArrays(mesh.draw_mode, 0, mesh.count)


def draw_limb(parent_matrix, start, end, thickness, quad_mesh, model_loc, color_loc, noise_loc, color, apply_noise=True):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx)) - 90.0
    mid = [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5]
    local_m = compose_transform_2d(tx=mid[0], ty=mid[1], angle_deg=angle, sx=thickness, sy=length)
    draw_shape(quad_mesh, parent_matrix @ local_m, color, model_loc, color_loc, noise_loc, apply_noise=apply_noise)


def draw_joint(parent_matrix, pos, radius, circle_mesh, model_loc, color_loc, noise_loc, color, apply_noise=True):
    local_m = compose_transform_2d(tx=pos[0], ty=pos[1], angle_deg=0.0, sx=radius * 2.0, sy=radius * 2.0)
    draw_shape(circle_mesh, parent_matrix @ local_m, color, model_loc, color_loc, noise_loc, apply_noise=apply_noise)


def draw_ground_fingers(parent_matrix, wrist_pos, dir_x, triangle_mesh, model_loc, color_loc, noise_loc, body_color):
    angles = [-35.0, -18.0, 0.0, 20.0] if dir_x < 0 else [35.0, 18.0, 0.0, -20.0]
    lengths = [0.22, 0.28, 0.32, 0.25]
    widths  = [0.035, 0.038, 0.040, 0.032]
    for ang, length, width in zip(angles, lengths, widths):
        local_m = compose_transform_2d(tx=wrist_pos[0], ty=wrist_pos[1], angle_deg=(180.0 + ang), sx=width, sy=length)
        draw_shape(triangle_mesh, parent_matrix @ local_m, body_color, model_loc, color_loc, noise_loc, apply_noise=True)


def draw_character(parent_matrix, quad_mesh, triangle_mesh, eye_mesh, circle_mesh, pupil_mesh, model_loc, color_loc, noise_loc):
    if wireframe_mode:
        body_color = np.array([0.0, 1.0, 0.8, 1.0], dtype=np.float32)   # Ciano vibrante
        eye_color  = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)   # Amarelo
        pupil_color= np.array([1.0, 0.2, 0.6, 1.0], dtype=np.float32)   # Magenta
    else:
        body_color = np.array([0.04, 0.04, 0.04, 1.0], dtype=np.float32)
        eye_color  = np.array([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
        pupil_color= np.array([0.04, 0.04, 0.04, 1.0], dtype=np.float32)

    # 1. Tronco e Bacia
    draw_limb(parent_matrix, [-0.16, 0.42], [0.14, 0.46], 0.065, quad_mesh, model_loc, color_loc, noise_loc, body_color)
    chest_m = compose_transform_2d(tx=-0.01, ty=0.30, angle_deg=-5.0, sx=0.16, sy=0.24)
    draw_shape(quad_mesh, parent_matrix @ chest_m, body_color, model_loc, color_loc, noise_loc, apply_noise=True)

    draw_limb(parent_matrix, [-0.01, 0.20], [0.05, -0.16], 0.11, quad_mesh, model_loc, color_loc, noise_loc, body_color)
    pelvis_m = compose_transform_2d(tx=0.05, ty=-0.16, angle_deg=10.0, sx=0.15, sy=0.13)
    draw_shape(quad_mesh, parent_matrix @ pelvis_m, body_color, model_loc, color_loc, noise_loc, apply_noise=True)

    # 2. Pescoço
    draw_limb(parent_matrix, [-0.01, 0.44], [-0.03, 0.72], 0.045, quad_mesh, model_loc, color_loc, noise_loc, body_color)

    # 3. Braços Longos (articulações reduzidas)
    l_shoulder, l_elbow, l_wrist = [-0.16, 0.42], [-0.25, -0.05], [-0.28, -0.72]
    draw_limb(parent_matrix, l_shoulder, l_elbow, 0.046, quad_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_joint(parent_matrix, l_elbow, 0.024, circle_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_limb(parent_matrix, l_elbow, l_wrist, 0.038, quad_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_joint(parent_matrix, l_wrist, 0.020, circle_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_ground_fingers(parent_matrix, l_wrist, -1.0, triangle_mesh, model_loc, color_loc, noise_loc, body_color)

    r_shoulder, r_elbow, r_wrist = [0.14, 0.46], [0.24, -0.02], [0.26, -0.74]
    draw_limb(parent_matrix, r_shoulder, r_elbow, 0.046, quad_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_joint(parent_matrix, r_elbow, 0.024, circle_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_limb(parent_matrix, r_elbow, r_wrist, 0.038, quad_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_joint(parent_matrix, r_wrist, 0.020, circle_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_ground_fingers(parent_matrix, r_wrist, 1.0, triangle_mesh, model_loc, color_loc, noise_loc, body_color)

    # 4. Pernas (articulações reduzidas)
    l_hip, l_knee, l_ankle = [0.01, -0.20], [-0.04, -0.55], [-0.02, -0.88]
    draw_limb(parent_matrix, l_hip, l_knee, 0.048, quad_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_joint(parent_matrix, l_knee, 0.025, circle_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_limb(parent_matrix, l_knee, l_ankle, 0.038, quad_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_joint(parent_matrix, l_ankle, 0.021, circle_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_limb(parent_matrix, l_ankle, [-0.08, -0.92], 0.045, quad_mesh, model_loc, color_loc, noise_loc, body_color)

    r_hip, r_knee, r_ankle = [0.09, -0.19], [0.10, -0.53], [0.08, -0.88]
    draw_limb(parent_matrix, r_hip, r_knee, 0.048, quad_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_joint(parent_matrix, r_knee, 0.025, circle_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_limb(parent_matrix, r_knee, r_ankle, 0.038, quad_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_joint(parent_matrix, r_ankle, 0.021, circle_mesh, model_loc, color_loc, noise_loc, body_color)
    draw_limb(parent_matrix, r_ankle, [0.02, -0.92], 0.045, quad_mesh, model_loc, color_loc, noise_loc, body_color)

    # 5. Cabeça e Espinhos
    head_center = [-0.03, 0.82]
    spike_angles = [0.0, 38.0, -38.0, 180.0, 142.0, 218.0, 90.0, 270.0]
    for angle in spike_angles:
        rad = math.radians(angle)
        pos = [head_center[0] - math.sin(rad) * 0.05, head_center[1] + math.cos(rad) * 0.05]
        spike_m = compose_transform_2d(tx=pos[0], ty=pos[1], angle_deg=angle, sx=0.14, sy=0.30)
        draw_shape(triangle_mesh, parent_matrix @ spike_m, body_color, model_loc, color_loc, noise_loc, apply_noise=True)

    head_base_m = compose_transform_2d(tx=head_center[0], ty=head_center[1], angle_deg=0.0, sx=0.28, sy=0.17)
    draw_shape(circle_mesh, parent_matrix @ head_base_m, body_color, model_loc, color_loc, noise_loc, apply_noise=True)

    eye_m = compose_transform_2d(tx=head_center[0], ty=head_center[1], angle_deg=0.0, sx=0.25, sy=0.12)
    draw_shape(eye_mesh, parent_matrix @ eye_m, eye_color, model_loc, color_loc, noise_loc, apply_noise=False)

    pupil_m = compose_transform_2d(tx=head_center[0], ty=head_center[1], angle_deg=0.0, sx=0.14, sy=0.12)
    draw_shape(pupil_mesh, parent_matrix @ pupil_m, pupil_color, model_loc, color_loc, noise_loc, apply_noise=True)


def draw_skeleton_body(parent_matrix, quad_mesh, circle_mesh, arc_mesh, spine_mesh, model_loc, color_loc, noise_loc):
    bone_color = np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32) if wireframe_mode else np.array([0.88, 0.85, 0.78, 1.0], dtype=np.float32)

    # Coluna
    draw_shape(spine_mesh, parent_matrix, bone_color, model_loc, color_loc, noise_loc, apply_noise=False)

    # Costelas
    rib_center_x, rib_center_y = 0.02, 0.35
    rib_width, rib_height, rib_count = 0.18, 0.25, 7
    for i in range(rib_count):
        offset_y = rib_center_y + (i * (rib_height / rib_count)) - (rib_height / 2)
        curve_w = rib_width * (1 - 0.3 * abs(i - rib_count / 2) / (rib_count / 2))
        rib_m = compose_transform_2d(tx=rib_center_x, ty=offset_y, angle_deg=0.0, sx=curve_w, sy=0.02)
        draw_shape(arc_mesh, parent_matrix @ rib_m, bone_color, model_loc, color_loc, noise_loc, apply_noise=False)

    # Pélvis
    pelvis = [0.0, 0.0]
    draw_joint(parent_matrix, pelvis, 0.06, circle_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)
    draw_limb(parent_matrix, [-0.05, -0.01], [0.03, -0.01], 0.012, quad_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)

    # Braço
    shoulder, elbow, wrist = [0.01, 0.50], [0.0, 0.28], [-0.05, 0.10]
    draw_joint(parent_matrix, shoulder, 0.02, circle_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)
    draw_joint(parent_matrix, elbow, 0.018, circle_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)
    draw_limb(parent_matrix, shoulder, elbow, 0.012, quad_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)
    draw_limb(parent_matrix, elbow, wrist, 0.010, quad_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)
    hand_end = [wrist[0] - 0.03, wrist[1] - 0.05]
    draw_limb(parent_matrix, wrist, hand_end, 0.008, quad_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)

    # Perna
    knee, ankle = [-0.30, 0.05], [-0.65, 0.0]
    draw_joint(parent_matrix, knee, 0.025, circle_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)
    draw_joint(parent_matrix, ankle, 0.02, circle_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)
    draw_limb(parent_matrix, pelvis, knee, 0.016, quad_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)
    draw_limb(parent_matrix, knee, ankle, 0.012, quad_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)
    foot_end = [ankle[0] - 0.12, ankle[1] - 0.02]
    draw_limb(parent_matrix, ankle, foot_end, 0.010, quad_mesh, model_loc, color_loc, noise_loc, bone_color, apply_noise=False)


def draw_skeleton_head(parent_matrix, circle_mesh, model_loc, color_loc, noise_loc):
    if wireframe_mode:
        bone_color   = np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32)
        socket_color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        bone_color   = np.array([0.88, 0.85, 0.78, 1.0], dtype=np.float32)
        socket_color = np.array([0.1, 0.1, 0.1, 1.0], dtype=np.float32)

    skull_m = compose_transform_2d(tx=0.0, ty=0.0, angle_deg=0.0, sx=0.16, sy=0.16)
    draw_shape(circle_mesh, parent_matrix @ skull_m, bone_color, model_loc, color_loc, noise_loc, apply_noise=False)

    jaw_m = compose_transform_2d(tx=-0.02, ty=-0.06, angle_deg=0.0, sx=0.06, sy=0.06)
    draw_shape(circle_mesh, parent_matrix @ jaw_m, bone_color, model_loc, color_loc, noise_loc, apply_noise=False)

    socket_m = compose_transform_2d(tx=-0.025, ty=0.01, angle_deg=0.0, sx=0.03, sy=0.05)
    draw_shape(circle_mesh, parent_matrix @ socket_m, socket_color, model_loc, color_loc, noise_loc, apply_noise=False)


def draw_skeleton(quad_mesh, circle_mesh, arc_mesh, spine_mesh, model_loc, color_loc, noise_loc):
    body_matrix = compose_transform_2d(
        tx=skel_state['tx'], ty=skel_state['ty'],
        angle_deg=skel_state['rotation'],
        sx=skel_state['scale'], sy=skel_state['scale']
    )

    draw_skeleton_body(body_matrix, quad_mesh, circle_mesh, arc_mesh, spine_mesh, model_loc, color_loc, noise_loc)

    if skel_head_state['falling']:
        head_matrix = compose_transform_2d(
            tx=skel_head_state['x'], ty=skel_head_state['y'],
            angle_deg=skel_head_state['rotation'], sx=1.0, sy=1.0
        )
    else:
        head_local = compose_transform_2d(tx=0.0, ty=0.65, angle_deg=0.0, sx=1.0, sy=1.0)
        head_matrix = body_matrix @ head_local

    draw_skeleton_head(head_matrix, circle_mesh, model_loc, color_loc, noise_loc)


SCENE_ASPECT = 500.0 / 900.0


def apply_letterboxed_viewport(width, height):
    if width <= 0 or height <= 0:
        return

    window_aspect = width / height
    if window_aspect > SCENE_ASPECT:
        vp_height = height
        vp_width = int(height * SCENE_ASPECT)
    else:
        vp_width = width
        vp_height = int(width / SCENE_ASPECT)

    vp_x = (width - vp_width) // 2
    vp_y = (height - vp_height) // 2
    glViewport(vp_x, vp_y, vp_width, vp_height)


def framebuffer_size_callback(window, width, height):
    apply_letterboxed_viewport(width, height)


# ==========================================
# MAIN LOOP
# ==========================================
def main():
    if not glfw.init():
        sys.exit()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    primary_monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(primary_monitor)
    screen_width, screen_height = video_mode.size.width, video_mode.size.height

    window = glfw.create_window(
        screen_width, screen_height,
        "Corredor + Cruz 3D + Monstro + Esqueleto",
        primary_monitor, None
    )
    if not window:
        glfw.terminate()
        sys.exit()

    # Contexto e Callbacks (Apenas UMA vez!)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_key_callback(window, key_event)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    fb_width, fb_height = glfw.get_framebuffer_size(window)
    apply_letterboxed_viewport(fb_width, fb_height)

    # 1. COMPILAÇÃO DOS SHADERS
    corridor_shader = compileProgram(
        compileShader(CORRIDOR_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(CORRIDOR_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    cross_shader = compileProgram(
        compileShader(CROSS_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(CROSS_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    char_shader = compileProgram(
        compileShader(CHAR_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(CHAR_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    # 2. MALHA E VAO DO CORREDOR
    corridor_verts = create_corridor_mesh()
    corridor_VAO, corridor_VBO = glGenVertexArrays(1), glGenBuffers(1)
    glBindVertexArray(corridor_VAO)
    glBindBuffer(GL_ARRAY_BUFFER, corridor_VBO)
    glBufferData(GL_ARRAY_BUFFER, corridor_verts.nbytes, corridor_verts, GL_STATIC_DRAW)
    stride_corr = 5 * corridor_verts.itemsize
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride_corr, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride_corr, ctypes.c_void_p(2 * corridor_verts.itemsize))
    glEnableVertexAttribArray(1)

    # 3. REAPROVEITA A CRUZ DEFINIDA EM cross.py
    from cross import create_cross_mesh as build_cross_mesh, MATRIZCORES_FACES as CROSS_COLORS
    cross_verts = build_cross_mesh(CROSS_COLORS)
    cross_VAO, cross_VBO = glGenVertexArrays(1), glGenBuffers(1)
    glBindVertexArray(cross_VAO)
    glBindBuffer(GL_ARRAY_BUFFER, cross_VBO)
    glBufferData(GL_ARRAY_BUFFER, cross_verts.nbytes, cross_verts, GL_STATIC_DRAW)
    
    stride_cross = 7 * cross_verts.itemsize  # 3 pos + 4 cor = 7 floats
    # Atributo 0: Posição (3 floats)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride_cross, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # Atributo 1: Cor (4 floats, offset de 3 floats = 12 bytes)
    offset_cor = 3 * cross_verts.itemsize
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride_cross, ctypes.c_void_p(offset_cor))
    glEnableVertexAttribArray(1)

    # 5. DEMAIS MALHAS COMPARTILHADAS E UNIFORMS
    quad_mesh     = create_quad()
    triangle_mesh = create_triangle()
    eye_mesh      = create_eye_shape(40)
    circle_mesh   = create_circle(64)
    pupil_mesh    = create_pupil_shape(40)

    arc_mesh   = create_arc_mesh(10)
    spine_mesh = create_spine_mesh(10)

    loc_cross_trans = glGetUniformLocation(cross_shader, "mat_transformation")

    c_model_loc = glGetUniformLocation(char_shader, "uModel")
    c_color_loc = glGetUniformLocation(char_shader, "uColor")
    c_noise_loc = glGetUniformLocation(char_shader, "uApplyNoise")

    print("\n--- CONTROLES ---")
    print("Cruz 3D:       SETAS move | Q/W/A/S/E/R gira | Z/X escala")
    print("Monstro:       I/K/J/L move | U/O gira | N/M escala")
    print("Esqueleto:     T/G/F/H move | Y/V gira | ,/. escala")
    print("Esqueleto:     C derruba a cabeça | 0 reseta o esqueleto")
    print("Wireframe (malhas visíveis e coloridas): P")
    print("Sair: ESC")
    print("-----------------\n")

    glBindVertexArray(0)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        update_skeleton_physics()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(2.5)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glLineWidth(1.0)

        # -------------------------------------------------------------
        # 1. RENDERIZA CORREDOR 2D
        # -------------------------------------------------------------
        glDisable(GL_DEPTH_TEST)
        glUseProgram(corridor_shader)
        glBindVertexArray(corridor_VAO)
        glDrawArrays(GL_TRIANGLES, 0, len(corridor_verts) // 5)

        # -------------------------------------------------------------
        # 2. RENDERIZA CRUZ 3D
        # -------------------------------------------------------------
        glEnable(GL_DEPTH_TEST)
        glUseProgram(cross_shader)

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
        for face_idx in range(16):
            glDrawArrays(GL_TRIANGLE_STRIP, face_idx * 4, 4)

        # -------------------------------------------------------------
        # 3. RENDERIZA MONSTRO 2D E ESQUELETO 2D (HIERÁRQUICOS)
        # -------------------------------------------------------------
        glDisable(GL_DEPTH_TEST)
        glUseProgram(char_shader)

        global_monster_matrix = compose_transform_2d(
            tx=char_pos[0], ty=char_pos[1],
            angle_deg=char_rotation,
            sx=char_scale[0], sy=char_scale[1]
        )
        draw_character(
            global_monster_matrix,
            quad_mesh, triangle_mesh, eye_mesh, circle_mesh, pupil_mesh,
            c_model_loc, c_color_loc, c_noise_loc
        )

        draw_skeleton(
            quad_mesh, circle_mesh, arc_mesh, spine_mesh,
            c_model_loc, c_color_loc, c_noise_loc
        )

        glfw.swap_buffers(window)

    glDeleteVertexArrays(1, [corridor_VAO])
    glDeleteVertexArrays(1, [cross_VAO])
    glDeleteBuffers(1, [corridor_VBO])
    glDeleteBuffers(1, [cross_VBO])
    glDeleteProgram(corridor_shader)
    glDeleteProgram(cross_shader)
    glDeleteProgram(char_shader)
    glfw.terminate()


if __name__ == "__main__":
    main()