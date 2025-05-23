import math
import numpy as np
class BinaryHeap:
    def __init__(self): #инициализация кучи
        self.heap = []

    def is_empty(self): #проверка на пустоту кучи
        return len(self.heap) == 0

    def insert(self, item): #добавление в кучу элементов
        self.heap.append(item)
        self.sift_up(len(self.heap) - 1)

    def pop(self): #извлечение элемента с самым высоким приоритетом
        if len(self.heap) == 0: #если длина кучи 0
            return None
        if len(self.heap) == 1: #если в куче 1 элемент, то делаем его возврат
            return self.heap.pop()

        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0] #меняем местами элементы кучи
        item = self.heap.pop()
        self.sift_down(0)
        return item #возвращаем кучу

    def sift_up(self, index): #восстановление приоритетности кучи снизу вверх
        parent = (index - 1) // 2
        if parent >= 0 and self.heap[index] < self.heap[parent]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            self.sift_up(parent)

    def sift_down(self, index):# восстановление приоритетности кучи сверху вниз
        left = 2 * index + 1
        right = 2 * index + 2
        smallest = index

        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self.sift_down(smallest)


class SortedArray:
    def __init__(self): #инициализация пустого массива для будущего отсортированного
        self.items = []

    def is_empty(self): #проверка на пустоту массива
        return len(self.items) == 0

    def insert(self, item): #добавление элемента чтобы не нарушить отсортированность его
        low, high = 0, len(self.items) #используем бинарный поиск нахождение позиции куда поставить элемент
        while low < high:
            mid = (low + high) // 2
            if self.items[mid].priority < item.priority:
                low = mid + 1
            else:
                high = mid
        self.items.insert(low, item)

    def pop(self): #извлечение элемента
        if self.is_empty(): #если массив пуст
            return None
        return self.items.pop(0) # иначе возвращаем самый первый элемент массива



class Search(): #класс, позволяющий хранить конкректное состояние доски
    def __init__(self, board, moves, previous, priority):
        self.board = board #текущее состояние доски
        self.moves = moves #количество сделанных ходов
        self.previous = previous #предыдущее состояние (ссылка)
        self.priority_function = priority #тип эвристики
        self.priority = self.calculate_priority() #общий приоритет

    def calculate_priority(self): #вычисление приоритета состояния
        if self.priority_function =='manhattan': #если тип эвристики будет Манхэттеновская
            return self.moves + self.board.manhattan()
        else: #тип эвристики будет Хэмминговская
            return self.moves + self.board.hamming()

    def __lt__(self, other): #сравнение состояний по приоритету
        return self.priority < other.priority

class Solver():
    def __init__(self, board, priority, heap_type='heap'):
        #Инициализация работы класса
        self.board = board #инициализация доски
        self.heap_type = heap_type #тип кучи
        self.priority = priority #инициализация типа эвристики
        self.solution = None # инициализация списка, содержащего последовательность решений
        self.solution_moves = -1 #инициализация количества шагов требуемых для решения
        self.solved = False #инициализация решаемости состояния
        self.search_nodes = 0 #инициализация счётчика состояний игры
        self.visited = {} # инициализация словаря для хранения состояний
        self.solve() # вызов метода реализованного агоритмом A*

    def isSolvable(self): #проверяет имеет ли начальное состояние решение
        return self.solved

    def moves(self): # возвращает количество шагов если решение имеется
        return self.solution_moves if self.solved else -1

    def solve(self): # основной метод реализации алгортим A*
        if self.board.is_goal(): #проверка на начальное состояние
            self.solved = True #решение есть
            self.solution_moves = 0 #количество шагов для решения 0
            self.solution = [self.board] #состояние доски окончательное
            return

        if self.heap_type == 'heap':
            HeapClass = BinaryHeap #Инициализация бинарной кучи
        else:
            HeapClass = SortedArray #Инициализация отсортированного массива

        main = HeapClass #основная куча
        twin = HeapClass #куча для проверки на решаемость

        main.insert(Search(self.board,0,None,self.priority)) #Добавление начальных состояний в основную кучу
        twin.insert(Search(self.board.twin(),0, None, self.priority)) #Добавление начальных состояний в дополнительную кучу

        while True: #основной цикл поиска
            if not main.is_empty(): #проверка, пуста ли куча
                current = main.pop() #достаём элемент кучи
                self.search_nodes += 1 #добавление счётчика состояния игры

                if current.board.is_goal(): #проверка на решение в основной куче
                    self.solved = True
                    self.solution_moves = current.moves()
                    self.solution = []
                    node = current
                    while node:
                        self.solution.insert(0, node.board)
                        node = node.previous()
                    return

                #цикл обрабатывающий соседей для основной кучи
                for x in current.board.neighbors():
                    neighbor_key = self.board_to_key(x)
                    if neighbor_key not in self.visited:
                        self.visited[neighbor_key] = True
                        new_node = Search(x, current.moves + 1, current, self.priority)
                        main.insert(new_node)

            #проверка очереди на решаемость
            if not twin.is_empty(): #если куча не пустая
                twin_cur = twin.pop()
                if twin_cur.board.is_goal():
                    self.solved = False
                    return

                # цикл обрабатывающий соседей
                for x in twin_cur.board.neighbors():
                    neighbor_key = self.board_to_key(x)
                    if neighbor_key not in self.visited:
                        self.visited[neighbor_key] = True
                        new_node = Search(x, twin_cur.moves + 1, twin_cur, self.priority)
                        main.insert(new_node)

    def board_to_key(self, board):#преобразование доски в ключ
        return tuple(tuple(row) for row in board.blocks)

    def __iter__(self):
        if self.solution:
            return iter(self.solution)
        return iter([])

    def __next__(self):
        if not hasattr(self, '_iter'):
            self._iter = iter(self.solution) if self.solution else iter([])
        return next(self._iter)


class Board:
    def __init__(self, blocks):  # создает поле из массива блоков NxN blocks (block[i][j] =
        # номер блока в i-той строке и j-том столбце)
        if isinstance(blocks, list): # проверяем, что мы получили на вход матрицу или список и создаём доску
            self.N = int(math.sqrt(len(blocks))) # определяем размерность доски
            self.board = np.ones((self.N, self.N)) # создаём матрицу
            self.board = self.board.astype(int)
            for i in range(self.N): # заполняем матрицу входными данными
                line = blocks[self.N * i:self.N * (i + 1)]
                self.board[i] = line
        if isinstance(blocks, np.ndarray): # если на вход получили матрицу, то просто копируем её в доску
            self.board = blocks.copy()
            self.N = blocks.shape[0] # определяем размерность доски
        # создаём итоговую доску
        self.target_board = np.arange(1, self.N * self.N + 1).reshape((self.N, self.N))
        self.target_board[-1, -1] = 0
        # определяем позицию пустого места
        i, j = np.where(self.board == 0)
        i = int(i[0])
        j = int(j[0])
        self.blank_pos = (i, j)


    def __str__(self):
        return str(self.board).replace('[', ' ').replace(']', ' ')

    def dimension(self):# возвращает размер доски N
        return self.N

    def hamming(self): # возвращает количество отличий от целевой доски
        mistakes = 0
        # для всех строк для всех элементов кроме последнего элемента последней строки сравниваем правильный элемент с элементов, который стоит в нашей доске и если он неправильный увеличиваем счетчик
        for i in range(self.N - 1):
            for j in range(self.N):
                if self.target_board[i][j] != self.board[i][j]:
                    mistakes += 1
        for j in range(self.N - 1):
            if self.target_board[self.N - 1][j] != self.board[self.N - 1][j]:
                mistakes += 1
        return mistakes

    def manhattan(self):
        length = 0
        # для всех строк для всех элементов кроме последнего элемента последней строки находим позицию элемента в нашей доске и ищем как далеко он олт правильного места
        for i in range(self.N - 1):
            for j in range(self.N):
                elem = self.target_board[i][j]
                col, row = np.where(self.board == elem) # находим позицию элемента
                length += abs(i - col[0]) + abs(j - row[0]) # добавляем в вывод расстояние до правильной позиции
        for j in range(self.N - 1):
            elem = self.target_board[self.N - 1][j]
            col, row = np.where(self.board == elem) # находим позицию элемента
            length += abs(self.N - 1 - col[0]) + abs(j - row[0]) # добавляем в вывод расстояние до правильной позиции

        return length

    # сумма манхэттенских расстояний блоков до целевых позиций

    def isGoal(self): # возвращает Истину, если текущее состояние целевое
        return np.array_equal(self.board, self.target_board)

    def twin(self): # меняет два соседних блока в строке и возвращает копию доски
        twin_board = self.board.copy() # копируем доску в новую переменную
        # ищем два непустых соседних элемента в строке
        for i in range(self.N):
            for j in range(self.N - 1):
                if twin_board[i, j] != 0 and twin_board[i, j + 1] != 0:
                    twin_board[i, j], twin_board[i, j + 1] = twin_board[i, j + 1], twin_board[i, j] # меняем местами
                    return Board(twin_board)

    def __eq__(self, board): # сравнивает две доски с помощью оператора сравнения
        return np.array_equal(self.board, board.board)


    def __iter__(self):
        self._neighbors = []
        i, j = self.blank_pos # позиция пустого места
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)] # все возможные ходы
        for i_move, j_move in moves:
            # ищем новую позицию элемента
            new_i = i + i_move
            new_j = j + j_move
            if 0 <= new_i < self.N and 0 <= new_j < self.N: # проверяем можем ли так походить
                new_board = self.board.copy() # создаем новую доску
                new_board[i, j], new_board[new_i, new_j] = new_board[new_i, new_j], new_board[i, j] # меняем местами пустое место и элемент
                self._neighbors.append(Board(new_board)) # добавляем новую доску в список соседей

        self._current = 0
        return self


    def __next__(self): # выводим соседей пока они есть
        if self._current >= len(self._neighbors):
            raise StopIteration
        result = self._neighbors[self._current]
        self._current += 1
        return result
