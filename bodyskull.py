import subprocess
import sys

# ==============================================================================
# MECANISMO DE INSTALAÇÃO AUTOMÁTICA
# Tenta importar os pacotes necessários. Se não encontrar, instala automaticamente.
# ==============================================================================
try:
    import OpenGL
    import pygame
except ImportError:
    print("Instalando pacotes necessários (pygame, PyOpenGL)...")
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
            ]
        )
        print("Instalação concluída com sucesso!\n")
    except Exception as e:
        print(f"\nErro na instalação automática: {e}")
        print("Por favor, tente instalar manualmente com o comando:")
        print("pip install pygame PyOpenGL PyOpenGL_accelerate")
        sys.exit(1)

# Importações principais
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *


def draw_circle(x, y, radius, num_segments=30):
    """Desenha um círculo preenchido para representação de juntas e crânio."""
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(num_segments + 1):
        angle = 2.0 * math.pi * i / num_segments
        dx = radius * math.cos(angle)
        dy = radius * math.sin(angle)
        glVertex2f(x + dx, y + dy)
    glEnd()


def draw_line(x1, y1, x2, y2, width=3):
    """Desenha uma linha espessa para ossos principais."""
    glLineWidth(width)
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()


def draw_ribs(center_x, center_y, width, height, count=7):
    """Desenha linhas curvas/paralelas representando a caixa torácica."""
    glLineWidth(2)
    for i in range(count):
        offset_y = center_y + (i * (height / count)) - (height / 2)
        curve_w = width * (1 - 0.3 * abs(i - count / 2) / (count / 2))

        # Desenha arcos simplificados para o tórax
        glBegin(GL_LINE_STRIP)
        for t in range(11):
            angle = math.pi * (t / 10.0)
            rx = center_x - (curve_w / 2) * math.cos(angle)
            ry = offset_y + 0.02 * math.sin(angle)
            glVertex2f(rx, ry)
        glEnd()


def draw_skeleton():
    """
    Renderiza o esqueleto com proporções reduzidas:
    - Coluna vertebral encurtada
    - Membros (braços e pernas) reduzidos proporcionalmente
    """
    # Cor do osso (branco acinzentado/creme)
    glColor3f(0.88, 0.85, 0.78)

    # ==========================================================
    # 1. Cabeça e Coluna Encurtada
    # ==========================================================
    head_x, head_y = 0.35, 0.50
    draw_circle(head_x, head_y, 0.075)  # Crânio reduzido
    draw_circle(head_x - 0.025, head_y - 0.06, 0.035)  # Mandíbula

    # Coluna Vertebral Encurtada (distância menor entre o topo e a pélvis)
    spine_top_x, spine_top_y = 0.37, 0.42
    pelvis_x, pelvis_y = 0.33, -0.15

    glLineWidth(4)
    glBegin(GL_LINE_STRIP)
    for i in range(11):
        t = i / 10.0
        sx = (1 - t) ** 2 * spine_top_x + 2 * (1 - t) * t * 0.40 + t**2 * pelvis_x
        sy = (1 - t) ** 2 * spine_top_y + 2 * (1 - t) * t * 0.15 + t**2 * pelvis_y
        glVertex2f(sx, sy)
    glEnd()

    # Caixa Torácica (Proporcionalmente menor)
    draw_ribs(center_x=0.36, center_y=0.18, width=0.20, height=0.26, count=7)

    # Bacia / Pélvis
    draw_circle(pelvis_x, pelvis_y, 0.065)
    draw_line(
        pelvis_x - 0.06, pelvis_y - 0.02, pelvis_x + 0.04, pelvis_y - 0.02, width=5
    )

    # ==========================================================
    # 2. Braço Encurtado e Rente ao Tronco
    # ==========================================================
    shoulder_x, shoulder_y = 0.35, 0.33
    elbow_x, elbow_y = 0.32, 0.08
    wrist_x, wrist_y = 0.20, -0.12

    # Articulações do Braço
    draw_circle(shoulder_x, shoulder_y, 0.025)
    draw_circle(elbow_x, elbow_y, 0.02)

    # Úmero (Braço Superior)
    draw_line(shoulder_x, shoulder_y, elbow_x, elbow_y, width=3)

    # Antebraço (Rádio e Ulna)
    draw_line(elbow_x, elbow_y + 0.008, wrist_x, wrist_y + 0.008, width=2)
    draw_line(elbow_x, elbow_y - 0.008, wrist_x, wrist_y - 0.008, width=2)

    # Mão / Dedos
    draw_line(wrist_x, wrist_y, wrist_x - 0.04, wrist_y - 0.06, width=2)

    # ==========================================================
    # 3. Pernas e Pés Encurtados
    # ==========================================================
    knee_x, knee_y = -0.05, -0.10
    ankle_x, ankle_y = -0.50, -0.18

    # Articulações da Perna
    draw_circle(knee_x, knee_y, 0.028)
    draw_circle(ankle_x, ankle_y, 0.02)

    # Fêmur
    draw_line(pelvis_x, pelvis_y, knee_x, knee_y, width=4)

    # Tíbia e Fíbula (canela)
    draw_line(knee_x, knee_y + 0.008, ankle_x, ankle_y + 0.008, width=2)
    draw_line(knee_x, knee_y - 0.008, ankle_x, ankle_y - 0.008, width=2)

    # Pés / Dedos estendidos
    draw_line(ankle_x, ankle_y, ankle_x - 0.15, ankle_y - 0.02, width=2)


def main():
    pygame.init()
    display = (800, 600)
    try:
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    except pygame.error:
        print("Erro: Não foi possível inicializar o modo OpenGL.")
        sys.exit(1)

    pygame.display.set_caption("Esqueleto 2D - Proporções Ajustadas")

    # Projeção Ortográfica
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Suavização de linhas (anti-aliasing)
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        glClear(GL_COLOR_BUFFER_BIT)
        glClearColor(0.1, 0.1, 0.1, 1.0)

        draw_skeleton()

        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":
    main()