import pygame
import heapq
import math

SQRT2 = math.sqrt(2)

ORTHO_COST = 10  # Costos para los lineales (ortogonales)
DIAG_COST  = 14  # Costos para diagonales

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos - A*")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)        
PURPLE = (159, 43, 104)       # Lista abierta (calculados)
GRIS_CLARO = (211, 211, 211)        # Lista cerrada (visitado)
NARANJA = (255, 165, 0)   # Inicio
PURPURA = (128, 0, 128)   # Fin
CORAL = (248, 131, 121)        # Camino final

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        # Nota: Este proyecto usa x=fila, y=col para mantener consistencia interna.
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        self.color = PURPLE

    def hacer_cerrado(self):
        self.color = GRIS_CLARO

    def hacer_camino(self):
        self.color = CORAL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    # Obtiene los nodos vecinos para poder usarlos para calcular g y f
    def actualizar_vecinos(self, grid):
        self.vecinos = []
        total_filas = self.total_filas
        fila = self.fila
        col = self.col
        
        # Helper: dentro de bounds y no es pared
        def libre(rr, cc):
            return 0 <= rr < total_filas and 0 <= cc < total_filas and not grid[rr][cc].es_pared()

        # Ortogonales
        if fila < total_filas - 1 and libre(fila + 1, col):
            self.vecinos.append(grid[fila + 1][col])  # abajo
        if fila > 0 and libre(fila - 1, col):
            self.vecinos.append(grid[fila - 1][col])  # arriba
        if col < total_filas - 1 and libre(fila, col + 1):
            self.vecinos.append(grid[fila][col + 1])  # derecha
        if col > 0 and libre(fila, col - 1):
            self.vecinos.append(grid[fila][col - 1])  # izquierda

        # Diagonales (siempre permitidas si no son paredes)
        if fila < total_filas - 1 and col < total_filas - 1 and libre(fila + 1, col + 1):
            self.vecinos.append(grid[fila + 1][col + 1])  # abajo-derecha
        if fila < total_filas - 1 and col > 0 and libre(fila + 1, col - 1):
            self.vecinos.append(grid[fila + 1][col - 1])  # abajo-izquierda
        if fila > 0 and col < total_filas - 1 and libre(fila - 1, col + 1):
            self.vecinos.append(grid[fila - 1][col + 1])  # arriba-derecha
        if fila > 0 and col > 0 and libre(fila - 1, col - 1):
            self.vecinos.append(grid[fila - 1][col - 1])  # arriba-izquierda    
# este y el de abajo dibujan la cuadricula
def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

# Dibuja las lineas para separar cada nodo(cuadro)
def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

# actualiza la pantalla por si realiza un calculo nuevamente
def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

# Obitiene la posion del cuadro que seleccione
def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos  # Nota: pos es (x, y), aquí se invierte a (y, x) para mapear a (fila, col)
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

# Calcula la distancia manhattan entre dos puntos para poder
# saber la distancia estimada a el final y en base a eso sacar
# la H ()
def heuristica(p1, p2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    # scaled diagonal distance using integer costs
    return DIAG_COST * min(dx, dy) + ORTHO_COST*abs(dx-dy) 

# Reconstruye el camino viendo de donde se calculo cada
# nodo (padre) hasta llegar al inicio para saber el camino mas corto (Linea verde)
def reconstruir_camino(came_from, current, draw, inicio):
    # Retrocede desde 'current' (fin) hasta el inicio
    while current in came_from:
        current = came_from[current]
        if current != inicio:
            current.hacer_camino()
        draw()

def a_estrella(draw, grid, inicio, fin):
    # Estructuras de A* (usando heapq como cola de prioridad
    open_heap = [] # Lista abierta guarda la F y el nodo
    contador = 0  # desempate en heap

    # Calcula la G
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0

    # Calcula la F (g+h)
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = heuristica(inicio.get_pos(), fin.get_pos())
    # aqui lo guarda 
    heapq.heappush(open_heap, (f_score[inicio], heuristica(inicio.get_pos(), fin.get_pos()), -g_score[inicio], contador, inicio))
    # para checar si el nodo esta en la lista abierta
    open_set_hash = {inicio}  

    while open_heap:
        # Permite cerrar la ventana mientras corre el algoritmo
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        # saca el nodo con el menor f_score para usarlo
        current = heapq.heappop(open_heap)[4]
        open_set_hash.discard(current)

        # si el nodo sacado ya es el final entonces solo
        # reconstruimos el camino final y pinta el camino
        if current == fin:
            reconstruir_camino(came_from, fin, draw, inicio)
            # Reafirma colores de inicio y fin
            inicio.hacer_inicio()
            fin.hacer_fin()
            return True

        # Explorar vecinos calcula la g , su h y su f
        for vecino in current.vecinos:
            # Determine if vecino is diagonal relative to current
            dr = abs(vecino.fila - current.fila)
            dc = abs(vecino.col - current.col)
            if dr == 1 and dc == 1:
                move_cost = DIAG_COST   # diagonal
            else:
                move_cost = ORTHO_COST  # orthogonal

            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score[vecino]:
                came_from[vecino] = current
                g_score[vecino] = tentative_g
                f_score[vecino] = tentative_g + heuristica(vecino.get_pos(), fin.get_pos())
                if vecino not in open_set_hash:
                    contador += 1
                    heapq.heappush(open_heap, (f_score[vecino], heuristica(vecino.get_pos(), fin.get_pos()), -g_score[vecino], contador, vecino))
                    open_set_hash.add(vecino)
                    if vecino != fin and vecino != inicio:
                        vecino.hacer_abierto()

        draw()
        
        if current != inicio and current != fin:
            current.hacer_cerrado()# marca como visitado

    return False  # No se encontró camino
# pintar los nodos y correr el algoritmo
def main(ventana, ancho):
    FILAS = 11
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    global came_from
    came_from = {}

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                if 0 <= fila < FILAS and 0 <= col < FILAS:
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()

                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()

                    elif nodo != fin and nodo != inicio:
                        nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                if 0 <= fila < FILAS and 0 <= col < FILAS:
                    nodo = grid[fila][col]
                    nodo.restablecer()
                    if nodo == inicio:
                        inicio = None
                    elif nodo == fin:
                        fin = None

            if event.type == pygame.KEYDOWN:
                # SPACE: ejecutar A* (si hay inicio y fin)
                if event.key == pygame.K_SPACE and inicio and fin:
                    # Preparar vecinos
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    came_from = {}
                    a_estrella(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)

                # C: limpiar tablero
                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)
                    came_from = {}

    pygame.quit()

if __name__ == "__main__":
    main(VENTANA, ANCHO_VENTANA)
