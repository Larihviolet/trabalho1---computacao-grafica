 import subprocess
import sys
import math

# ==============================================================================
# MECANISMO DE INSTALAÇÃO AUTOMÁTICA
# Tenta importar os pacotes necessários. Se não encontrar, instala automaticamente.
# ==============================================================================
try:
    import OpenGL
    import pygame
    import numpy as np  # Adicionado para manipulação de matrizes
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
        # Reinicia o próprio script para carregar as novas bibliotecas instaladas
        subprocess.call([sys.executable] + sys.argv)
        sys.exit(0)
    except Exception as install_error:
        print(f"\nErro na instalação automática: {install_error}")
        print("Por favor, tente instalar manualmente com o comando:")
        print("pip install pygame PyOpenGL PyOpenGL_accelerate numpy")
        sys.exit(1)

# Importações principais do sistema e gráficas
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *


# ==============================================================================
# SEÇÃO 1: MATEMÁTICA DE MATRIZES MANUAIS (SEM MODEL/CAMERA)
# ============================================================================== 

wireframe_mode = False

def get_identity_matrix():
    """Retorna uma matriz identidade 4x4 padrão."""
    return np.identity(4, dtype=np.float32)

def get_translation_matrix(tx, ty, tz=0.0):
    """Cria manualmente uma matriz de translação 4x4."""
    matrix = get_identity_matrix()
    matrix[3, 0] = tx  # Coluna 4, Linha 1
    matrix[3, 1] = ty  # Coluna 4, Linha 2
    matrix[3, 2] = tz  # Coluna 4, Linha 3 (mantemos 0 para 2D)
    return matrix

def get_rotation_z_matrix(theta_rad):
    """Cria manualmente uma matriz de rotação 4x4 ao redor do eixo Z."""
    matrix = get_identity_matrix()
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    
    # Rotação Z padrão em 2D
    matrix[0, 0] = cos_t
    matrix[0, 1] = sin_t
    matrix[1, 0] = -sin_t
    matrix[1, 1] = cos_t
    return matrix

def get_scaling_matrix(sx, sy, sz=1.0):
    """Cria manualmente uma matriz de escala 4x4."""
    matrix = get_identity_matrix()
    matrix[0, 0] = sx  # Escala X
    matrix[1, 1] = sy  # Escala Y
    matrix[2, 2] = sz  # Escala Z (mantemos 1 para 2D)
    return matrix


# ==============================================================================
# SEÇÃO 2: DESENHO GEOMÉTRICO DO ESQUELETO 2D
# ==============================================================================

def draw_circle(x, y, radius, num_segments=20):
    """Desenha um círculo preenchido simples."""
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(num_segments + 1):
        angle = 2.0 * math.pi * i / num_segments
        dx = radius * math.cos(angle)
        dy = radius * math.sin(angle)
        glVertex2f(x + dx, y + dy)
    glEnd()

def draw_ellipse(x, y, radius_x, radius_y, num_segments=20):
    """Desenha uma forma oval preenchida (olho)."""
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(num_segments + 1):
        angle = 2.0 * math.pi * i / num_segments
        dx = radius_x * math.cos(angle)
        dy = radius_y * math.sin(angle)
        glVertex2f(x + dx, y + dy)
    glEnd()

def draw_line(x1, y1, x2, y2, width=3):
    """Desenha uma linha espessa (ossos principais)."""
    glLineWidth(width)
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()

def draw_ribs(center_x, center_y, width, height, count=7):
    """Desenha linhas curvas simplificadas (costelas)."""
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
    """Define apenas a geometria 'crua' do esqueleto ao redor da origem (0,0)."""
    # Cor do osso
    glColor3f(0.88, 0.85, 0.78)

    # --- Cabeça ---
    # Posicionada ligeiramente acima da origem
    head_x, head_y = 0.0, 0.65
    draw_circle(head_x, head_y, 0.08)  # Crânio
    draw_circle(head_x - 0.02, head_y - 0.06, 0.03)  # Mandíbula

    # Cavidade Ocular
    glColor3f(0.1, 0.1, 0.1)  # Escuro
    draw_ellipse(head_x - 0.025, head_y + 0.01, radius_x=0.015, radius_y=0.025)
    glColor3f(0.88, 0.85, 0.78)  # Volta cor osso

    # --- Coluna ---
    spine_top_x, spine_top_y = 0.02, 0.58
    pelvis_x, pelvis_y = 0.0, 0.0 # Pélvis na origem

    glLineWidth(3)
    glBegin(GL_LINE_STRIP)
    for i in range(11):
        t = i / 10.0
        # Curva Bezier simples da coluna
        sx = (1 - t) ** 2 * spine_top_x + 2 * (1 - t) * t * 0.10 + t**2 * pelvis_x
        sy = (1 - t) ** 2 * spine_top_y + 2 * (1 - t) * t * 0.30 + t**2 * pelvis_y
        glVertex2f(sx, sy)
    glEnd()

    # Costelas
    draw_ribs(center_x=0.02, center_y=0.35, width=0.18, height=0.25, count=7)

    # Pélvis (Bacia)
    draw_circle(pelvis_x, pelvis_y, 0.06)
    draw_line(pelvis_x - 0.05, pelvis_y - 0.01, pelvis_x + 0.03, pelvis_y - 0.01, width=4)

    # --- Braço (ao lado do tronco) ---
    shoulder_x, shoulder_y = 0.01, 0.50
    elbow_x, elbow_y = 0.0, 0.28
    wrist_x, wrist_y = -0.05, 0.10

    draw_circle(shoulder_x, shoulder_y, 0.02) # Ombro
    draw_circle(elbow_x, elbow_y, 0.018) # Cotovelo
    draw_line(shoulder_x, shoulder_y, elbow_x, elbow_y, width=3) # Úmero
    # Antebraço
    draw_line(elbow_x, elbow_y + 0.005, wrist_x, wrist_y + 0.005, width=2)
    draw_line(elbow_x, elbow_y - 0.005, wrist_x, wrist_y - 0.005, width=2)
    # Mão
    draw_line(wrist_x, wrist_y, wrist_x - 0.03, wrist_y - 0.05, width=2)

    # --- Pernas (esticadas) ---
    # Origem da pélvis
    knee_x, knee_y = -0.30, 0.05
    ankle_x, ankle_y = -0.65, 0.0

    draw_circle(knee_x, knee_y, 0.025) # Joelho
    draw_circle(ankle_x, ankle_y, 0.02) # Tornozelo
    draw_line(pelvis_x, pelvis_y, knee_x, knee_y, width=4) # Fêmur
    # Canela
    draw_line(knee_x, knee_y + 0.005, ankle_x, ankle_y + 0.005, width=2)
    draw_line(knee_x, knee_y - 0.005, ankle_x, ankle_y - 0.005, width=2)
    # Pé
    draw_line(ankle_x, ankle_y, ankle_x - 0.12, ankle_y - 0.02, width=2)


# ==========================================================
# SEÇÃO 3: SISTEMA DE CONTROLE E MATRIZ COMPOSTA
# ==========================================================

# Estado global das transformações que o usuário controla
state = {
    'tx': 0.0,    # Translação X
    'ty': 0.0,    # Translação Y
    'rotation': 0.0, # Ângulo em graus
    'scale': 1.0     # Fator de escala uniforme
}

def handle_input():
    """Processa teclas para alterar o estado das transformações."""
    keys = pygame.key.get_pressed()
    
    # Velocidades de movimento
    move_speed = 0.01
    rot_speed = 1.0
    scale_speed = 0.01

    # Translação: Setas
    if keys[K_LEFT]:  state['tx'] -= move_speed
    if keys[K_RIGHT]: state['tx'] += move_speed
    if keys[K_UP]:    state['ty'] += move_speed
    if keys[K_DOWN]:  state['ty'] -= move_speed

    # Rotação: A / D
    if keys[K_a]: state['rotation'] += rot_speed
    if keys[K_d]: state['rotation'] -= rot_speed

    # Escala: W (Aumentar) / S (Diminuir)
    if keys[K_w]: state['scale'] += scale_speed
    if keys[K_s]: state['scale'] -= scale_speed
    
    # Limita escala para não ficar negativa ou zerada
    state['scale'] = max(0.1, state['scale'])

    # Resetar: R
    if keys[K_r]:
        state['tx'] = 0.0
        state['ty'] = 0.0
        state['rotation'] = 0.0
        state['scale'] = 1.0


def draw_scene():
    """Gera a matriz composta e aplica ao OpenGL."""
    # 1. Cria as matrizes primárias manuais baseadas no estado controlado pelo usuário
    T = get_translation_matrix(state['tx'], state['ty'])
    
    # Converte rotação para radianos
    rot_rad = math.radians(state['rotation'])
    R = get_rotation_z_matrix(rot_rad)
    
    S = get_scaling_matrix(state['scale'], state['scale'])

    # 2. Compõe a matriz final (Multiplicação: Translação * Rotação * Escala)
    # A ordem importa muito! Nós multiplicamos usando o operador '@' do Numpy.
    composed_matrix = T @ R @ S

    # 3. Restrição: Não usamos glTranslatef/glRotatef/glScalef.
    # Em vez disso, carregamos nossa matriz composta DIRETAMENTE na MODELVIEW.
    glMatrixMode(GL_MODELVIEW)
    
    # OpenGL espera a matriz no formato 'column-major'. O Numpy cria 'row-major'.
    # Precisamos transpor (.T) ou carregar com a função glLoadMatrixf que aceita Numpy.
    # Usaremos glLoadMatrixf com a matriz transposta para garantir compatibilidade.
    glLoadMatrixf(composed_matrix.T)

    # 4. Desenha a geometria fixa
    draw_skeleton_geometry()


def main():
    pygame.init()
    display = (800, 600)
    try:
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    except pygame.error:
        print("Erro: Modo OpenGL não suportado.")
        sys.exit(1)

    pygame.display.set_caption("Controle Manual de Matriz 2D (T*R*S) - OpenGL")

    # Projeção Ortográfica Fixa
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)

    # Ativa suavização
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    # Imprime instruções
    print("\n--- CONTROLES ---")
    print("Mover: SETAS (Cima, Baixo, Esquerda, Direita)")
    print("Girar: A / D")
    print("Escala: W (Aumentar) / S (Diminuir)")
    print("Resetar: R")
    print("-----------------\n")

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 1. Processa entrada do usuário
        handle_input()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.1, 0.1, 0.1, 1.0)

        # 2. Gera matriz composta e desenha
        draw_scene()

        pygame.display.flip()
        clock.tick(60) # Limita a 60 FPS


if __name__ == "__main__":
    main()   
    