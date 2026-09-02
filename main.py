# Trabalho 1 - SCC0250 Computação Gráfica 
# Nome: Larissa Rocha Goncalves NUSP: 15522431
# Cena de Terror. Sendo os objetos:

   #1. Corredor 2D  - plano de fundo estático (chão, paredes, teto, porta).
   #2. Cômoda 2D    - móvel desenhado com quads/triângulos coloridos, com translação/rotação/escala controláveis pelo usuário.
   #3. Cruz 3D      - objeto 3D composto por 16 faces coloridas, controlado por translação e rotação nos 3 eixos.
   #4. Luminária 3D - disco/cone gerado proceduralmente (frente, aro e fundo), controlado apenas por escala.
   #5. Monstro 2D   - personagem articulado (hierárquico) feito de membros, juntas e triângulos, com "ruído" de cor aplicado via shader.
   #6. Esqueleto 2D - outro personagem hierárquico, cuja cabeça pode ser "derrubada" e sofre uma simulação simples de física
    #(gravidade, quique no chão e rotação/deslizamento).
 
# Arquitetura geral:
   #- Cada objeto tem seu próprio par de shaders (vertex/fragment) e função(ões) de geração de geometria (`create_*_mesh` / `build_*_geometry`).
   #- Transformações 2D (translação, rotação, escala) são compostas em uma única matriz 4x4 pela função `compose_transform_2d`.
   #- Objetos hierárquicos (monstro e esqueleto) são desenhados por composição de matrizes: cada "osso"/parte é desenhado multiplicando a matriz do pai pela transformação local da parte (ver `draw_limb`, `draw_joint`,
    #`draw_character`, `draw_skeleton*`).
   #- O estado de cada objeto controlável fica em dicionários/globais no topo
      #do arquivo (`cabinet_state`, `lamp_state`, `skel_state`, etc.) e é alterado pelos callbacks de teclado (`key_event`).
   #- O loop principal (`main`) compila os shaders, sobe as malhas para a
      # GPU, e a cada frame: processa eventos, atualiza física, limpa a tela e desenha cada objeto na ordem correta (2D sem depth test, depois 3D com depth test, depois os personagens 2D por cima).
import sys
import math
import ctypes
import subprocess

# Instalando os pacotes necessários
try:
    import numpy as np
    import glfw
    from OpenGL.GL import *
    from OpenGL.GL.shaders import compileProgram, compileShader
    # Se qualquer uma das libs acima não estiver instalada, tenta instalar
    # automaticamente via pip e reinicia o próprio script.
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

# Shaders do Cenário 2D (Corredor)

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

# Shaders da Cruz 3D
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

# Shaders da Luminária
LAMP_VERTEX_SRC = """
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

LAMP_FRAGMENT_SRC = """
#version 330 core
in vec3 fragColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(fragColor, 1.0);
}
"""

# Shaders da Cômoda 
CABINET_VERTEX_SRC = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
uniform mat4 uTransform;
out vec3 vColor;

void main() {
    gl_Position = uTransform * vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
"""

CABINET_FRAGMENT_SRC = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
"""


# Shaders de objetos (Monstro e Esqueleto)
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


# Variáveis de estado e controle
#Cada bloco abaixo guarda o estado atual (posição/rotação/escala) de um
# objeto da cena. Esses valores são lidos a cada frame no loop principal
# para montar a matriz de transformação do objeto, e são alterados pelos
# callbacks de teclado em `key_event`.

# Cruz 3D
t_x, t_y, t_z = -0.86, 0.1, 0.0
ang_x, ang_y, ang_z = -0.15, -0.5854, 0.15
s_factor = 0.22

# Monstro 2D
char_pos = [0.0, -0.05]
char_rotation = 0.0
char_scale = [0.12, 0.12]

wireframe_mode = False

# --- Cômoda 2D ---
# Estado controlável pelo usuário: translação (tx, ty), rotação e escala uniforme. 
cabinet_state = {
    'tx': -0.10,
    'ty': -0.11,
    'rotation': 0.0,
    'scale': 0.48,
}

# --- Luminária 3D ---

lamp_state = {
    'scale': 0.70,
}

# --- Esqueleto 2D ---
skel_state = {
    'tx': 0.78,
    'ty': -0.72,
    'rotation': 0.0,
    'scale': 1.0,
}
# Estado da simulação de física da cabeça do esqueleto quando ela "cai".
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

SKEL_GROUND_Y = -0.95 # altura considerada chão para a cabeça do esqueleto (quando atinge o chão, ela para de cair)
SKEL_GRAVITY = -0.0016 # aceleração vertical aplicada por frame
SKEL_LEFT_BOUND = -0.98 #limite esquerdo até onde a cabeça pode rolar

#Inicia a animação de queda da cabeça do esqueleto.
#Calcula a posição mundial atual da cabeça (aplicando a transformação do corpo do esqueleto: translação, rotação e escala) e usa esse pontocomo posição inicial da cabeça "solta", que passa a ser simulada
#independentemente do corpo em `update_skeleton_physics`.
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

# Restaura o esqueleto (corpo e cabeça) para o estado/posição inicial.
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

# Restaura a cômoda para sua posição, rotação e escala iniciais
def reset_cabinet():
    cabinet_state['tx'] = 0.0
    cabinet_state['ty'] = -0.12
    cabinet_state['rotation'] = 0.0
    cabinet_state['scale'] = 0.78

 #Comportamento, enquanto `skel_head_state['falling']` for True:
     # 1. Aplica gravidade à velocidade vertical (até tocar o chão).
     # 2. Integra a posição (x, y) pela velocidade atual.
     # 3. Detecta colisão com o "chão" (SKEL_GROUND_Y): ao tocar, zera a velocidade vertical e define uma rotação alvo (3 voltas completas)
     # para simular a cabeça rolando.
     # 4. Enquanto a rotação atual for menor que a rotação alvo, 
     #gira a cabeça e a desloca horizontalmente de forma proporcional ao raio(rolamento sem deslizar), 
     #com velocidade angular decrescente conforme se aproxima do alvo .
# 5. Impede que a cabeça ultrapasse o limite esquerdo da cena (SKEL_LEFT_BOUND), parando o movimento horizontal nesse ponto.
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

# Gera os vértices da Luminária 3D.
def generate_lamp_geometry(segments=64):
    vertices = []

    color_light = [1.0, 0.98, 0.88]
    color_rim = [1.0, 1.0, 1.0]
    color_back = [1.0, 1.0, 0.4]

    vertices.extend([0.0, 0.0, 0.2, *color_light])
    for i in range(segments + 1):
        angle = (2.0 * math.pi * i) / segments
        x = math.cos(angle)
        y = math.sin(angle)
        vertices.extend([x, y, 0.2, *color_light])

    rim_depth = -0.12  # Profundidade do aro branco (era -0.3; valor menos negativo = aro mais curto/fino)

    rim_start = segments + 2
    for i in range(segments + 1):
        angle = (2.0 * math.pi * i) / segments
        x = math.cos(angle)
        y = math.sin(angle)
        vertices.extend([x, y, 0.2, *color_rim])
        vertices.extend([x, y, rim_depth, *color_rim])

    back_start = rim_start + (segments + 1) * 2
    vertices.extend([0.0, 0.0, rim_depth, *color_back])
    for i in range(segments + 1):
        angle = (2.0 * math.pi * i) / segments
        x = math.cos(angle)
        y = math.sin(angle)
        vertices.extend([x, y, rim_depth, *color_back])

    return np.array(vertices, dtype=np.float32), segments, rim_start, back_start


def create_lamp_mesh():
    vertices, segments, rim_start, back_start = generate_lamp_geometry()
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    stride = 6 * vertices.itemsize
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)

    return vao, vbo, segments, rim_start, back_start


def add_quad(vertices, p1, p2, p3, p4, color):
    r, g, b = color
    pts = [p1, p2, p3, p1, p3, p4]
    for x, y in pts:
        vertices.extend([x, y, r, g, b])


def add_circle(vertices, cx, cy, rx, ry, color, segments=16):
    r, g, b = color
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    for i in range(segments):
        a1 = angles[i]
        a2 = angles[(i + 1) % segments]
        p1 = (cx, cy)
        p2 = (cx + rx * math.cos(a1), cy + ry * math.sin(a1))
        p3 = (cx + rx * math.cos(a2), cy + ry * math.sin(a2))
        for x, y in [p1, p2, p3]:
            vertices.extend([x, y, r, g, b])

    # Monta, "à mão" (coordenadas fixas), toda a geometria 2D da cômoda com
    # espelho, usando `add_quad` e `add_circle`.
    # A cômoda é construída em partes, todas em coordenadas locais (antes de
    # qualquer translação/rotação/escala global, que é aplicada depois via
    # `cabinet_state` + `compose_transform_2d`):
    # - Espelho: moldura externa, moldura interna e vidro (glass), com um
    #   deslocamento vertical (`mirror_offset_y`) para ficar acima do corpo.
    # - Corpo/estrutura: topo, lados (com painel "inset") e base com uma
    #   pequena espessura para simular profundidade.
    # - Frente com gavetas: a face frontal é dividida em 3 linhas x 2
    #   colunas de gavetas. Para cada célula, é feito um contorno + um
    #   painel interno ligeiramente menor (efeito de baixo-relevo) e 2
    #   maçanetas, com interpolação bilinear das coordenadas Y ao longo
    #   da largura (`interp_y`) para acompanhar a perspectiva/trapézio da frente do móvel.
def build_cabinet_geometry():
    vertices = []
    c_outline = (0.06, 0.04, 0.02)
    c_top = (0.42, 0.27, 0.15)
    c_side = (0.30, 0.18, 0.10)
    c_side_inset = (0.22, 0.13, 0.07)
    c_front = (0.35, 0.21, 0.12)
    c_drawer = (0.38, 0.24, 0.14)
    c_knob = (0.85, 0.65, 0.20)
    c_mirror_frame = (0.28, 0.17, 0.09)
    c_glass = (0.42, 0.54, 0.56)
    mirror_offset_y = 0.32

    add_quad(vertices, (-0.88, 0.62 + mirror_offset_y), (-0.42, 0.38 + mirror_offset_y), (-0.42, -0.05 + mirror_offset_y), (-0.88, 0.10 + mirror_offset_y), c_outline)
    add_quad(vertices, (-0.86, 0.59 + mirror_offset_y), (-0.44, 0.36 + mirror_offset_y), (-0.44, -0.03 + mirror_offset_y), (-0.86, 0.12 + mirror_offset_y), c_mirror_frame)
    add_quad(vertices, (-0.80, 0.52 + mirror_offset_y), (-0.48, 0.32 + mirror_offset_y), (-0.48, 0.05 + mirror_offset_y), (-0.80, 0.19 + mirror_offset_y), c_outline)
    add_quad(vertices, (-0.79, 0.50 + mirror_offset_y), (-0.49, 0.31 + mirror_offset_y), (-0.49, 0.07 + mirror_offset_y), (-0.79, 0.20 + mirror_offset_y), c_glass)

    s_bl = (-0.93, -0.85)
    s_br = (-0.44, -0.88)
    s_tr = (-0.44, 0.16)
    s_tl = (-0.93, 0.19)
    t_back_right = (-0.10, -0.01)
    add_quad(vertices, s_tl, s_tr, t_back_right, (-0.60, 0.02), c_outline)
    add_quad(vertices, s_tl, (-0.44, 0.15), (-0.11, -0.02), (-0.59, 0.03), c_top)
    add_quad(vertices, s_tl, s_tr, s_br, s_bl, c_outline)
    add_quad(vertices, (-0.92, 0.18), (-0.45, 0.15), (-0.45, -0.87), (-0.92, -0.84), c_side)
    add_quad(vertices, (-0.86, 0.11), (-0.52, 0.09), (-0.52, -0.74), (-0.86, -0.72), c_outline)
    add_quad(vertices, (-0.85, 0.10), (-0.53, 0.08), (-0.53, -0.73), (-0.85, -0.71), c_side_inset)

    c_thick = 0.012
    add_quad(vertices, (-0.93, -0.85), (-0.44, -0.88), (-0.44, -0.88 - c_thick), (-0.93, -0.85 - c_thick), c_outline)

    f_tl = (-0.44, 0.16)
    f_tr = (-0.10, -0.01)
    f_br = (-0.10, -0.46)
    f_bl = (-0.44, -0.88)
    add_quad(vertices, f_tl, f_tr, f_br, f_bl, c_outline)
    add_quad(vertices, (-0.43, 0.15), (-0.11, -0.02), (-0.11, -0.45), (-0.43, -0.87), c_front)

    col_split_x = -0.26
    front_left_x = -0.43
    front_right_x = -0.11
    y_lines = [(0.13, -0.04), (-0.16, -0.16), (-0.47, -0.28), (-0.78, -0.41)]

    # Interpolação linear da altura Y (topo/base de uma gaveta) entre a
    # borda esquerda e a borda direita da frente da cômoda, dado um x
    # intermediário — usado para desenhar a divisória central das
    # gavetas seguindo o mesmo trapézio da frente do móvel.
    def interp_y(x, y_l, y_r):
        t = (x - front_left_x) / (front_right_x - front_left_x)
        return y_l + t * (y_r - y_l)
        
    for i in range(3):
        # Para cada uma das 3 linhas de gavetas, calcula os 4 cantos
                # (topo-esquerda/direita, base-esquerda/direita) e desenha duas
                # gavetas (esquerda e direita), cada uma com contorno, painel
                # interno e 2 maçanetas.
        yt_l, yt_r = y_lines[i]
        yb_l, yb_r = y_lines[i + 1]
        yt_m = interp_y(col_split_x, yt_l, yt_r)
        yb_m = interp_y(col_split_x, yb_l, yb_r)

        gx1_l, gx1_r = front_left_x + 0.01, col_split_x - 0.008
        gy1_tl = yt_l - 0.015
        gy1_tr = yt_m - 0.015
        gy1_br = yb_m + 0.015
        gy1_bl = yb_l + 0.015
        add_quad(vertices, (gx1_l, gy1_tl), (gx1_r, gy1_tr), (gx1_r, gy1_br), (gx1_l, gy1_bl), c_outline)
        add_quad(vertices, (gx1_l + 0.005, gy1_tl - 0.005), (gx1_r - 0.005, gy1_tr - 0.005), (gx1_r - 0.005, gy1_br + 0.005), (gx1_l + 0.005, gy1_bl + 0.005), c_drawer)
        for u in [0.28, 0.72]:
            kx = gx1_l + u * (gx1_r - gx1_l)
            ky_top = gy1_tl + u * (gy1_tr - gy1_tl)
            ky_bot = gy1_bl + u * (gy1_br - gy1_bl)
            ky = (ky_top + ky_bot) * 0.5
            rx = 0.010 - 0.004 * (kx - front_left_x) / (front_right_x - front_left_x)
            ry = 0.015 - 0.005 * (kx - front_left_x) / (front_right_x - front_left_x)
            add_circle(vertices, kx, ky, rx, ry, c_knob)

        gx2_l, gx2_r = col_split_x + 0.008, front_right_x - 0.01
        gy2_tl = yt_m - 0.015
        gy2_tr = yt_r - 0.015
        gy2_br = yb_r + 0.015
        gy2_bl = yb_m + 0.015
        add_quad(vertices, (gx2_l, gy2_tl), (gx2_r, gy2_tr), (gx2_r, gy2_br), (gx2_l, gy2_bl), c_outline)
        add_quad(vertices, (gx2_l + 0.005, gy2_tl - 0.005), (gx2_r - 0.005, gy2_tr - 0.005), (gx2_r - 0.005, gy2_br + 0.005), (gx2_l + 0.005, gy2_bl + 0.005), c_drawer)
        for u in [0.28, 0.72]:
            kx = gx2_l + u * (gx2_r - gx2_l)
            ky_top = gy2_tl + u * (gy2_tr - gy2_tl)
            ky_bot = gy2_bl + u * (gy2_br - gy2_bl)
            ky = (ky_top + ky_bot) * 0.5
            rx = 0.010 - 0.004 * (kx - front_left_x) / (front_right_x - front_left_x)
            ry = 0.015 - 0.005 * (kx - front_left_x) / (front_right_x - front_left_x)
            add_circle(vertices, kx, ky, rx, ry, c_knob)

    return np.array(vertices, dtype=np.float32)

# Constrói a geometria da cômoda (`build_cabinet_geometry`) e a envia para a GPU em um VAO/VBO.
def create_cabinet_mesh():
    vertices = build_cabinet_geometry()
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    stride = 5 * vertices.itemsize
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * vertices.itemsize))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)
    return vao, vbo, len(vertices) // 5

# Callback de teclado do GLFW: interpreta teclas pressionadas/seguradas
# e atualiza o estado (posição/rotação/escala) dos objetos da cena.

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
        if key == glfw.KEY_9:
            reset_cabinet()
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

        # --- Controles da Cômoda 2D (translação e escala) ---
        if key == glfw.KEY_D: cabinet_state['tx'] += 0.01
        if key == glfw.KEY_B: cabinet_state['tx'] -= 0.01
        if key == glfw.KEY_EQUAL: cabinet_state['ty'] += 0.01
        if key == glfw.KEY_MINUS: cabinet_state['ty'] -= 0.01

        if key == glfw.KEY_RIGHT_BRACKET:
            cabinet_state['scale'] = min(3.0, cabinet_state['scale'] + 0.01)
        if key == glfw.KEY_LEFT_BRACKET:
            cabinet_state['scale'] = max(0.05, cabinet_state['scale'] - 0.01)

        if key == glfw.KEY_APOSTROPHE: cabinet_state['rotation'] += 1.0
        if key == glfw.KEY_SEMICOLON:  cabinet_state['rotation'] -= 1.0

        # --- Controles da Luminária 3D (escala) ---
        if key == glfw.KEY_8: lamp_state['scale'] = min(3.0, lamp_state['scale'] + 0.02)
        if key == glfw.KEY_7: lamp_state['scale'] = max(0.05, lamp_state['scale'] - 0.02)


# Álgebra de matrizes 4x4
    #Monta uma matriz de transformação 4x4 (linha-maior, para uso com
    #glUniformMatrix4fv(..., GL_TRUE, ...)) equivalente a:
    #Translação(tx, ty) * Rotação(angle_deg) * Escala(sx, sy)
    # aplicada a pontos 2D "embutidos" em 4D (z=0, w=1). É a função central
    #usada para posicionar praticamente todos os objetos 2D da cena
    #(membros do monstro/esqueleto, cômoda, ossos etc.).
    #tx, ty: translação nos eixos X e Y.
    # angle_deg: rotação em graus, sentido anti-horário.
    # sx, sy: fatores de escala nos eixos X e Y. 
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

# Estrutura simples que agrupa os dados necessários para desenhar uma
#malha: o VAO já configurado na GPU, a contagem de vértices e o modo
# de desenho (GL_TRIANGLES, GL_TRIANGLE_FAN, GL_LINE_STRIP, etc.)
class Mesh:
    def __init__(self, vao, count, draw_mode=GL_TRIANGLES):
        self.vao = vao
        self.count = count
        self.draw_mode = draw_mode

# Função auxiliar interna
def _upload_2d_mesh(vertices, draw_mode):
    vertices = np.array(vertices, dtype=np.float32)
    vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return vao, len(vertices) // 2


#Gera os vértices (posição XY + cor RGB) do corredor de fundo: chão,
#duas paredes laterais, teto e uma "porta" central mais escura ao
#fundo, todos desenhados como triângulos em coordenadas NDC fixas
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

#Gera os vértices (posição XYZ + cor RGBA) da cruz 3D a partir de uma
# lista fixa de 64 pontos brutos (16 faces x 4 vértices cada, organizados como GL_TRIANGLE_STRIP por face).
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
#Cria um quad 1x1 centrado na origem (dois triângulos), sem cor por vértice
def create_quad():
    vertices = [-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5]
    vao, count = _upload_2d_mesh(vertices, GL_TRIANGLES)
    return Mesh(vao, count, GL_TRIANGLES)

#Cria um triângulo isósceles apontando para cima, centrado na origem
def create_triangle():
    vertices = [0.0, 1.0, -0.5, 0.0, 0.5, 0.0]
    vao, count = _upload_2d_mesh(vertices, GL_TRIANGLES)
    return Mesh(vao, count, GL_TRIANGLES)

# Cria a forma da pupila (um "losango" com bordas curvas, feito de dois
    #arcos de cosseno opostos), desenhado como GL_TRIANGLE_FAN a partir do
    #centro (0, 0).
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

# Cria a forma do olho (mesma ideia da pupila, mas rotacionada 90°:
    #curvas de cosseno no eixo Y em vez de X), desenhado como
    #GL_TRIANGLE_FAN a partir do centro (0, 0).
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

#"Cria um círculo preenchido de raio 0.5, como GL_TRIANGLE_FAN a partir do centro
def create_circle(segments=48):
    vertices = [0.0, 0.0]
    for i in range(segments + 1):
        theta = 2.0 * math.pi * float(i) / float(segments)
        vertices.extend([math.cos(theta) * 0.5, math.sin(theta) * 0.5])
    vao, count = _upload_2d_mesh(vertices, GL_TRIANGLE_FAN)
    return Mesh(vao, count, GL_TRIANGLE_FAN)

#Cria um arco semicircular (meia elipse) como GL_LINE_STRIP, usado para
    #desenhar as costelas do esqueleto (cada costela é este mesmo arco,
    #reescalado/reposicionado
def create_arc_mesh(segments=10):
    vertices = []
    for i in range(segments + 1):
        t = i / segments
        x = -0.5 * math.cos(t * math.pi)
        y = math.sin(t * math.pi)
        vertices.extend([x, y])
    vao, count = _upload_2d_mesh(vertices, GL_LINE_STRIP)
    return Mesh(vao, count, GL_LINE_STRIP)

#Cria a coluna vertebral do esqueleto como uma curva de Bézier
    # quadrática (do topo da coluna até a pélvis, passando por um ponto de
    #controle), desenhada como GL_LINE_STRIP
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



# RENDERIZAÇÃO DE OBJETOS HIERÁRQUICOS
#  Desenha uma única `Mesh` com uma matriz de modelo e cor específicas,
# usando o shader "de personagem" (CHAR_VERTEX_SRC/CHAR_FRAGMENT_SRC)
def draw_shape(mesh, model_matrix, color, model_loc, color_loc, noise_loc, apply_noise=True):
    glUniformMatrix4fv(model_loc, 1, GL_TRUE, model_matrix.flatten())
    glUniform4fv(color_loc, 1, color)
    glUniform1i(noise_loc, 1 if (apply_noise and not wireframe_mode) else 0)
    glBindVertexArray(mesh.vao)
    glDrawArrays(mesh.draw_mode, 0, mesh.count)

# Desenha um "membro" (osso) como um quad alongado e rotacionado, ligando dois pontos `start` e `end` no espaço local do pai
def draw_limb(parent_matrix, start, end, thickness, quad_mesh, model_loc, color_loc, noise_loc, color, apply_noise=True):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx)) - 90.0
    mid = [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5]
    local_m = compose_transform_2d(tx=mid[0], ty=mid[1], angle_deg=angle, sx=thickness, sy=length)
    draw_shape(quad_mesh, parent_matrix @ local_m, color, model_loc, color_loc, noise_loc, apply_noise=apply_noise)

# Desenha uma "junta" (articulação) como um círculo de raio `radius`
# centrado em `pos`, no espaço local do pai — usado para suavizar as
# conexões entre membros (cotovelos, joelhos, tornozelos, pélvis etc.
def draw_joint(parent_matrix, pos, radius, circle_mesh, model_loc, color_loc, noise_loc, color, apply_noise=True):
    local_m = compose_transform_2d(tx=pos[0], ty=pos[1], angle_deg=0.0, sx=radius * 2.0, sy=radius * 2.0)
    draw_shape(circle_mesh, parent_matrix @ local_m, color, model_loc, color_loc, noise_loc, apply_noise=apply_noise)

#Desenha 4 "dedos" (triângulos alongados) irradiando a partir do
#pulso (`wrist_pos`) do monstro, em ângulos e comprimentos fixos, para
#simular uma mão apoiada no chão.
def draw_ground_fingers(parent_matrix, wrist_pos, dir_x, triangle_mesh, model_loc, color_loc, noise_loc, body_color):
    angles = [-35.0, -18.0, 0.0, 20.0] if dir_x < 0 else [35.0, 18.0, 0.0, -20.0]
    lengths = [0.22, 0.28, 0.32, 0.25]
    widths  = [0.035, 0.038, 0.040, 0.032]
    for ang, length, width in zip(angles, lengths, widths):
        local_m = compose_transform_2d(tx=wrist_pos[0], ty=wrist_pos[1], angle_deg=(180.0 + ang), sx=width, sy=length)
        draw_shape(triangle_mesh, parent_matrix @ local_m, body_color, model_loc, color_loc, noise_loc, apply_noise=True)

#Desenha 4 "dedos" (triângulos alongados) irradiando a partir do
#pulso (`wrist_pos`) do monstro, em ângulos e comprimentos fixos, para
#simular uma mão apoiada no chão.

#Ordem de desenho (todas coordenadas em espaço local do personagem):
    #1. Tronco e bacia (membros + preenchimentos de peito/pélvis).
    #2. Pescoço.
    #3. Braços longos (ombro -> cotovelo -> pulso, com juntas arredondadas e "dedos" no chão em cada pulso).
    #4. Pernas (quadril -> joelho -> tornozelo -> pé).
    #5. Cabeça: espinhos ao redor (8 triângulos em ângulos fixos),base da cabeça (círculo), olho (forma de "amêndoa") e pupila.
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

#Desenha o corpo do esqueleto (tudo exceto a cabeça, que é tratada
    #separadamente para poder "cair" de forma independente): coluna,
   # costelas, pélvis, um braço (ombro->cotovelo->pulso->mão) e uma perna
   # (quadril->joelho->tornozelo->pé)
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

# Desenha a caveira do esqueleto: crânio (círculo grande), mandíbula (círculo menor, deslocado para baixo) e uma "órbita ocular"
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

#Monta a matriz global do corpo a partir de `skel_state` e desenha o
   #corpo com `draw_skeleton_body`. Para a cabeça, há dois casos:
   #- Se `skel_head_state['falling']` for True: a cabeça é desenhada
    # em sua posição/rotação mundial simulada por `update_skeleton_physics` (independente do corpo).
   #- Caso contrário: a cabeça é desenhada "presa" ao corpo, usando uma transformação local fixa (0, 0.65) composta com a matrizdo corpo.
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

# Proporção fixa (largura/altura) da cena "de referência" (500x900),
# usada para calcular um viewport com letterbox/pillarbox e manter essa
# proporção independentemente do tamanho real da janela.
SCENE_ASPECT = 500.0 / 900.0

# Calcula e aplica um viewport centralizado que preserva a proporção SCENE_ASPECT, adicionando barras pretas (letterbox/pillarbox) se necessário.
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

# Main Loop
def main():
    if not glfw.init():
        sys.exit()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(
        1100, 800,
        "Corredor + Cruz 3D + Monstro + Esqueleto",
        None, None
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

    # 1. Compilação dos shaders 
    corridor_shader = compileProgram(
        compileShader(CORRIDOR_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(CORRIDOR_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    cross_shader = compileProgram(
        compileShader(CROSS_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(CROSS_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    lamp_shader = compileProgram(
        compileShader(LAMP_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(LAMP_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    cabinet_shader = compileProgram(
        compileShader(CABINET_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(CABINET_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    char_shader = compileProgram(
        compileShader(CHAR_VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(CHAR_FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )

    # 2. Malha e vao do corredor 2D
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

    # 3. Reaproveita a cruz definida em cross.py
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

    # 5. LUMINÁRIA 3D
    lamp_vao, lamp_vbo, lamp_segments, lamp_rim_start, lamp_back_start = create_lamp_mesh()
    lamp_transform_loc = glGetUniformLocation(lamp_shader, "uTransform")

    # 6. CÔMODA 2D
    cabinet_vao, cabinet_vbo, cabinet_count = create_cabinet_mesh()
    cabinet_transform_loc = glGetUniformLocation(cabinet_shader, "uTransform")

    # 7. DEMAIS MALHAS COMPARTILHADAS E UNIFORMS
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
    print("Cômoda:        D/B move X | =/- move Y | ]/[ escala | '/; gira | 9 reseta a cômoda")
    print("Luminária:     7/8 diminui/aumenta escala")
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

        
        # 1. RENDERIZA CORREDOR 2D
        glDisable(GL_DEPTH_TEST)
        glUseProgram(corridor_shader)
        glBindVertexArray(corridor_VAO)
        glDrawArrays(GL_TRIANGLES, 0, len(corridor_verts) // 5)

        # 2. RENDERIZA CÔMODA 2D
        glDisable(GL_DEPTH_TEST)
        glUseProgram(cabinet_shader)

        # Matriz composta a partir das transformações geométricas primárias
        # (translação, rotação e escala), controláveis pelo usuário via cabinet_state.
        cabinet_matrix = compose_transform_2d(
            tx=cabinet_state['tx'], ty=cabinet_state['ty'],
            angle_deg=cabinet_state['rotation'],
            sx=cabinet_state['scale'], sy=cabinet_state['scale']
        )
        glUniformMatrix4fv(cabinet_transform_loc, 1, GL_TRUE, cabinet_matrix.flatten())

        glBindVertexArray(cabinet_vao)
        glDrawArrays(GL_TRIANGLES, 0, cabinet_count)

        
        # 3. RENDERIZA CRUZ 3D
        glEnable(GL_DEPTH_TEST)
        glUseProgram(cross_shader)

        mat_escala = np.array([
            [s_factor, 0.0, 0.0, 0.0],
            [0.0, s_factor, 0.0, 0.0],
            [0.0, 0.0, s_factor, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
         # Rotação em torno do eixo X
        cos_x, sin_x = math.cos(ang_x), math.sin(ang_x)
        mat_rx = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_x, -sin_x, 0.0],
            [0.0, sin_x, cos_x, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        # Rotação em torno do eixo Y
        cos_y, sin_y = math.cos(ang_y), math.sin(ang_y)
        mat_ry = np.array([
            [cos_y, 0.0, sin_y, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_y, 0.0, cos_y, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        # Rotação em torno do eixo Z 
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
        # Composição final: T * (Rz * Ry * Rx) * S — rotações combinadas
        # (ordem Z, depois Y, depois X) antes de escalar e transladar.
        mat_composta_cross = mat_translacao @ (mat_rz @ mat_ry @ mat_rx) @ mat_escala
        glUniformMatrix4fv(loc_cross_trans, 1, GL_TRUE, mat_composta_cross.flatten())

        glBindVertexArray(cross_VAO)
        for face_idx in range(16):
            glDrawArrays(GL_TRIANGLE_STRIP, face_idx * 4, 4)

        
        # 3. RENDERIZA LUMINÁRIA 3D
        glUseProgram(lamp_shader)
        lamp_S = np.array([
            [0.25 * lamp_state['scale'], 0.0, 0.0, 0.0],
            [0.0, 0.25 * lamp_state['scale'], 0.0, 0.0],
            [0.0, 0.0, 0.12 * lamp_state['scale'], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        lamp_R_x = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, math.cos(math.radians(-72)), -math.sin(math.radians(-72)), 0.0],
            [0.0, math.sin(math.radians(-72)), math.cos(math.radians(-72)), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        lamp_T = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.78],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        lamp_matrix = lamp_T @ lamp_R_x @ lamp_S
        glUniformMatrix4fv(lamp_transform_loc, 1, GL_TRUE, lamp_matrix.flatten())
        glBindVertexArray(lamp_vao)
        glDrawArrays(GL_TRIANGLE_FAN, lamp_back_start, lamp_segments + 2)
        glDrawArrays(GL_TRIANGLE_STRIP, lamp_rim_start, (lamp_segments + 1) * 2)
        glDrawArrays(GL_TRIANGLE_FAN, 0, lamp_segments + 2)

       
        # 4. RENDERIZA MONSTRO 2D E ESQUELETO 2D (HIERÁRQUICOS)
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

    # Liberação de recursos da GPU ao encerrar o programa.
    glDeleteVertexArrays(1, [corridor_VAO])
    glDeleteVertexArrays(1, [cross_VAO])
    glDeleteVertexArrays(1, [lamp_vao])
    glDeleteVertexArrays(1, [cabinet_vao])
    glDeleteBuffers(1, [corridor_VBO])
    glDeleteBuffers(1, [cross_VBO])
    glDeleteBuffers(1, [lamp_vbo])
    glDeleteBuffers(1, [cabinet_vbo])
    glDeleteProgram(corridor_shader)
    glDeleteProgram(cross_shader)
    glDeleteProgram(lamp_shader)
    glDeleteProgram(cabinet_shader)
    glDeleteProgram(char_shader)
    glfw.terminate()


if __name__ == "__main__":
    main()