import math
import subprocess
import sys

# ==============================================================================
# MECANISMO DE INSTALAÇÃO AUTOMÁTICA
# ==============================================================================
try:
    import numpy as np
    import OpenGL
    import pygame
except ImportError as e:
    print(f"Pacote faltando detectado: {e.name}. Instalando dependências...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "pygame",
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
        print("Por favor, instale manualmente: pip install pygame PyOpenGL PyOpenGL_accelerate numpy")
        sys.exit(1)

from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

# ==============================================================================
# SEÇÃO 1: MATEMÁTICA DE MATRIZES MANUAIS
# ==============================================================================

wireframe_mode = False  # Controle global do estado da malha


def get_identity_matrix():
    return np.identity(4, dtype=np.float32)


def get_translation_matrix(tx, ty, tz=0.0):
    matrix = get_identity_matrix()
    matrix[0, 3] = tx
    matrix[1, 3] = ty
    matrix[2, 3] = tz
    return matrix


def get_rotation_z_matrix(theta_rad):
    matrix = get_identity_matrix()
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    matrix[0, 0] = cos_t
    matrix[0, 1] = -sin_t
    matrix[1, 0] = sin_t
    matrix[1, 1] = cos_t
    return matrix


def get_scaling_matrix(sx, sy, sz=1.0):
    matrix = get_identity_matrix()
    matrix[0, 0] = sx
    matrix[1, 1] = sy
    matrix[2, 2] = sz
    return matrix


# ==============================================================================
# SEÇÃO 2: DESENHO GEOMÉTRICO DO ESQUELETO 2D
# ==============================================================================


def draw_circle(x, y, radius, num_segments=20):
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(num_segments + 1):
        angle = 2.0 * math.pi * i / num_segments
        dx = radius * math.cos(angle)
        dy = radius * math.sin(angle)
        glVertex2f(x + dx, y + dy)
    glEnd()


def draw_ellipse(x, y, radius_x, radius_y, num_segments=20):
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(num_segments + 1):
        angle = 2.0 * math.pi * i / num_segments
        dx = radius_x * math.cos(angle)
        dy = radius_y * math.sin(angle)
        glVertex2f(x + dx, y + dy)
    glEnd()


def draw_line(x1, y1, x2, y2, width=3):
    glLineWidth(width)
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()


def draw_ribs(center_x, center_y, width, height, count=7):
    glLineWidth(1)
    for i in range(count):
        offset_y = center_y + (i * (height / count)) - (height / 2)
        curve_w = width * (1 - 0.3 * abs(i - count / 2) / (count / 2))

        glBegin(GL_LINE_STRIP)
        for t in range(11):
            angle = math.pi * (t / 10.0)
            rx = center_x - (curve_w / 2) * math.cos(angle)
            ry = offset_y + 0.02 * math.sin(angle)
            glVertex2f(rx, ry)
        glEnd()


def draw_skeleton_geometry():
    glColor3f(0.88, 0.85, 0.78)

    # Cabeça
    head_x, head_y = 0.0, 0.65
    draw_circle(head_x, head_y, 0.08)
    draw_circle(head_x - 0.02, head_y - 0.06, 0.03)

    # Cavidade Ocular
    glColor3f(0.1, 0.1, 0.1)
    draw_ellipse(head_x - 0.025, head_y + 0.01, radius_x=0.015, radius_y=0.025)
    glColor3f(0.88, 0.85, 0.78)

    # Coluna
    spine_top_x, spine_top_y = 0.02, 0.58
    pelvis_x, pelvis_y = 0.0, 0.0

    glLineWidth(3)
    glBegin(GL_LINE_STRIP)
    for i in range(11):
        t = i / 10.0
        sx = (1 - t) ** 2 * spine_top_x + 2 * (1 - t) * t * 0.10 + t**2 * pelvis_x
        sy = (1 - t) ** 2 * spine_top_y + 2 * (1 - t) * t * 0.30 + t**2 * pelvis_y
        glVertex2f(sx, sy)
    glEnd()

    # Costelas
    draw_ribs(center_x=0.02, center_y=0.35, width=0.18, height=0.25, count=7)

    # Pélvis
    draw_circle(pelvis_x, pelvis_y, 0.06)
    draw_line(pelvis_x - 0.05, pelvis_y - 0.01, pelvis_x + 0.03, pelvis_y - 0.01, width=4)

    # Braço
    shoulder_x, shoulder_y = 0.01, 0.50
    elbow_x, elbow_y = 0.0, 0.28
    wrist_x, wrist_y = -0.05, 0.10

    draw_circle(shoulder_x, shoulder_y, 0.02)
    draw_circle(elbow_x, elbow_y, 0.018)
    draw_line(shoulder_x, shoulder_y, elbow_x, elbow_y, width=3)
    draw_line(elbow_x, elbow_y + 0.005, wrist_x, wrist_y + 0.005, width=2)
    draw_line(elbow_x, elbow_y - 0.005, wrist_x, wrist_y - 0.005, width=2)
    draw_line(wrist_x, wrist_y, wrist_x - 0.03, wrist_y - 0.05, width=2)

    # Pernas
    knee_x, knee_y = -0.30, 0.05
    ankle_x, ankle_y = -0.65, 0.0

    draw_circle(knee_x, knee_y, 0.025)
    draw_circle(ankle_x, ankle_y, 0.02)
    draw_line(pelvis_x, pelvis_y, knee_x, knee_y, width=4)
    draw_line(knee_x, knee_y + 0.005, ankle_x, ankle_y + 0.005, width=2)
    draw_line(knee_x, knee_y - 0.005, ankle_x, ankle_y - 0.005, width=2)
    draw_line(ankle_x, ankle_y, ankle_x - 0.12, ankle_y - 0.02, width=2)


# ==========================================================
# SEÇÃO 3: SISTEMA DE CONTROLE E MATRIZ COMPOSTA
# ==========================================================

state = {
    'tx': 0.0,
    'ty': 0.0,
    'rotation': 0.0,
    'scale': 1.0,
}


def handle_input():
    keys = pygame.key.get_pressed()

    move_speed = 0.01
    rot_speed = 1.0
    scale_speed = 0.01

    if keys[K_LEFT]:
        state['tx'] -= move_speed
    if keys[K_RIGHT]:
        state['tx'] += move_speed
    if keys[K_UP]:
        state['ty'] += move_speed
    if keys[K_DOWN]:
        state['ty'] -= move_speed

    if keys[K_a]:
        state['rotation'] += rot_speed
    if keys[K_d]:
        state['rotation'] -= rot_speed

    if keys[K_w]:
        state['scale'] += scale_speed
    if keys[K_s]:
        state['scale'] -= scale_speed

    state['scale'] = max(0.1, state['scale'])

    if keys[K_r]:
        state['tx'] = 0.0
        state['ty'] = 0.0
        state['rotation'] = 0.0
        state['scale'] = 1.0


def draw_scene():
    # Aplica o modo wireframe ou preenchido conforme a variável
    if wireframe_mode:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    # 1. Composição de Matrizes
    T = get_translation_matrix(state['tx'], state['ty'])
    R = get_rotation_z_matrix(math.radians(state['rotation']))
    S = get_scaling_matrix(state['scale'], state['scale'])
    composed_matrix = T @ R @ S

    # 2. Carrega na MODELVIEW
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(np.ascontiguousarray(composed_matrix.T, dtype=np.float32))

    # 3. Desenho
    draw_skeleton_geometry()


def main():
    global wireframe_mode
    pygame.init()
    display = (800, 600)
    try:
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    except pygame.error:
        print("Erro: Modo OpenGL não suportado.")
        sys.exit(1)

    pygame.display.set_caption("Controle Manual 2D (T*R*S) - P para Alternar Wireframe")

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)

    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    print("\n--- CONTROLES ---")
    print("Mover: SETAS (Cima, Baixo, Esquerda, Direita)")
    print("Girar: A / D")
    print("Escala: W (Aumentar) / S (Diminuir)")
    print("Alternar Malha/Wireframe: P")
    print("Resetar: R")
    print("-----------------\n")

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                # Alterna o modo wireframe ao pressionar P
                if event.key == pygame.K_p:
                    wireframe_mode = not wireframe_mode

        handle_input()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.1, 0.1, 0.1, 1.0)

        draw_scene()

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()