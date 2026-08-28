import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math
import ctypes

# ==========================================
# 1. Configurações da Janela e Proporção
# ==========================================
largura = 1000
altura = 700
aspect = largura / altura

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(largura, altura, "Transformacoes 3D Interativas", None, None)

if not window:
    print("Falha ao inicializar janela GLFW")
    glfw.terminate()
    raise RuntimeError("Erro ao criar janela")
    
glfw.make_context_current(window)

# ==========================================
# 2. Shaders
# ==========================================
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

program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
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
    raise RuntimeError(glGetProgramInfoLog(program).decode())

glUseProgram(program)

# ==========================================
# 3. Vértices do Cubo
# ==========================================
vertices = np.zeros(24, [("position", np.float32, 3)])
vertices['position'] = [
    # Face 1 (Frente)
    (-0.2, -0.2, +0.2), (+0.2, -0.2, +0.2), (-0.2, +0.2, +0.2), (+0.2, +0.2, +0.2),
    # Face 2 (Direita)
    (+0.2, -0.2, +0.2), (+0.2, -0.2, -0.2), (+0.2, +0.2, +0.2), (+0.2, +0.2, -0.2),
    # Face 3 (Trás)
    (+0.2, -0.2, -0.2), (-0.2, -0.2, -0.2), (+0.2, +0.2, -0.2), (-0.2, +0.2, -0.2),
    # Face 4 (Esquerda)
    (-0.2, -0.2, -0.2), (-0.2, -0.2, +0.2), (-0.2, +0.2, -0.2), (-0.2, +0.2, +0.2),
    # Face 5 (Baixo)
    (-0.2, -0.2, -0.2), (+0.2, -0.2, -0.2), (-0.2, -0.2, +0.2), (+0.2, -0.2, +0.2),
    # Face 6 (Cima)
    (-0.2, +0.2, +0.2), (+0.2, +0.2, +0.2), (-0.2, +0.2, -0.2), (+0.2, +0.2, -0.2)
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

# ==========================================
# 4. Variáveis de Estado das Transformações
# ==========================================
# Translação (x, y, z)
t_x = 0.0
t_y = 0.0
t_z = 0.0

# Rotação em radianos (x, y, z)
theta_x = math.radians(20) # leve inclinação inicial para 3D
theta_y = math.radians(30)
theta_z = 0.0

# Escala (x, y, z)
s_x = 2.5   # Alongado horizontalmente
s_y = 0.6   # Achatado verticalmente
s_z = 1.5   # Profundidade

def key_event(window, key, scancode, action, mods):
    global t_x, t_y, t_z, theta_x, theta_y, theta_z, s_x, s_y, s_z
    
    if action == glfw.PRESS or action == glfw.REPEAT:
        # Translação
        if key == glfw.KEY_LEFT:   t_x -= 0.05
        if key == glfw.KEY_RIGHT:  t_x += 0.05
        if key == glfw.KEY_UP:     t_y += 0.05
        if key == glfw.KEY_DOWN:   t_y -= 0.05
        
        # Rotação
        if key == glfw.KEY_W: theta_x += 0.1
        if key == glfw.KEY_S: theta_x -= 0.1
        if key == glfw.KEY_D: theta_y += 0.1
        if key == glfw.KEY_A: theta_y -= 0.1
        if key == glfw.KEY_E: theta_z += 0.1
        if key == glfw.KEY_Q: theta_z -= 0.1
        
        # Escala nos eixos individuais
        if key == glfw.KEY_U: s_x += 0.1 # Alongar X
        if key == glfw.KEY_J: s_x = max(0.1, s_x - 0.1) # Achatar X
        
        if key == glfw.KEY_I: s_y += 0.1 # Alongar Y
        if key == glfw.KEY_K: s_y = max(0.1, s_y - 0.1) # Achatar Y
        
        if key == glfw.KEY_O: s_z += 0.1 # Alongar Z
        if key == glfw.KEY_L: s_z = max(0.1, s_z - 0.1) # Achatar Z
        
        # Escala Uniforme
        if key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD: 
            s_x += 0.1; s_y += 0.1; s_z += 0.1
        if key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
            s_x = max(0.1, s_x - 0.1); s_y = max(0.1, s_y - 0.1); s_z = max(0.1, s_z - 0.1)

glfw.set_key_callback(window, key_event)

# ==========================================
# 5. Funções de Matrizes Primárias
# ==========================================
def matriz_translacao(tx, ty, tz):
    return np.array([
        1.0, 0.0, 0.0,  tx,
        0.0, 1.0, 0.0,  ty,
        0.0, 0.0, 1.0,  tz,
        0.0, 0.0, 0.0, 1.0
    ], np.float32)

def matriz_escala(sx, sy, sz):
    # Correção do aspect ratio na largura para manter a proporção da janela
    return np.array([
        sx / aspect, 0.0, 0.0, 0.0,
        0.0,          sy, 0.0, 0.0,
        0.0,         0.0,  sz, 0.0,
        0.0,         0.0, 0.0, 1.0
    ], np.float32)

def matriz_rotacao_x(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        1.0, 0.0, 0.0, 0.0,
        0.0,   c,  -s, 0.0,
        0.0,   s,   c, 0.0,
        0.0, 0.0, 0.0, 1.0
    ], np.float32)

def matriz_rotacao_y(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
          c, 0.0,   s, 0.0,
        0.0, 1.0, 0.0, 0.0,
         -s, 0.0,   c, 0.0,
        0.0, 0.0, 0.0, 1.0
    ], np.float32)

def matriz_rotacao_z(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
          c,  -s, 0.0, 0.0,
          s,   c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ], np.float32)

def multiplica_matriz(a, b):
    m_a = a.reshape(4, 4)
    m_b = b.reshape(4, 4)
    return np.dot(m_a, m_b).reshape(1, 16)

# ==========================================
# 6. Exibição e Laço Principal
# ==========================================
glfw.show_window(window)
glEnable(GL_DEPTH_TEST)

try:
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)    
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
        # 1. Obtenção das matrizes primárias
        T  = matriz_translacao(t_x, t_y, t_z)
        Rx = matriz_rotacao_x(theta_x)
        Ry = matriz_rotacao_y(theta_y)
        Rz = matriz_rotacao_z(theta_z)
        S  = matriz_escala(s_x, s_y, s_z)
        
        # 2. Composição: M = T * (Rz * (Ry * (Rx * S)))
        # Os vértices passam primeiro pela Escala, depois Rotações, depois Translação
        M = multiplica_matriz(Rx, S)
        M = multiplica_matriz(Ry, M)
        M = multiplica_matriz(Rz, M)
        mat_transform = multiplica_matriz(T, M)

        glUniformMatrix4fv(loc_transformation, 1, GL_TRUE, mat_transform) 
        
        # Face 1 (Vermelho)
        glUniform4f(loc_color, 1.0, 0.0, 0.0, 1.0)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        # Face 2 (Azul)
        glUniform4f(loc_color, 0.0, 0.0, 1.0, 1.0)
        glDrawArrays(GL_TRIANGLE_STRIP, 4, 4)
        
        # Face 3 (Verde)
        glUniform4f(loc_color, 0.0, 1.0, 0.0, 1.0)
        glDrawArrays(GL_TRIANGLE_STRIP, 8, 4)
        
        # Face 4 (Amarelo)
        glUniform4f(loc_color, 1.0, 1.0, 0.0, 1.0)
        glDrawArrays(GL_TRIANGLE_STRIP, 12, 4)
        
        # Face 5 (Cinza)
        glUniform4f(loc_color, 0.5, 0.5, 0.5, 1.0)
        glDrawArrays(GL_TRIANGLE_STRIP, 16, 4)
        
        # Face 6 (Marrom)
        glUniform4f(loc_color, 0.5, 0.0, 0.0, 1.0)
        glDrawArrays(GL_TRIANGLE_STRIP, 20, 4)
        
        glfw.swap_buffers(window)
        glfw.poll_events()

finally:
    glfw.destroy_window(window)
    glfw.terminate()