import math
import numpy as np
import random
import matplotlib.pyplot as plt
import time

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
        if self.board.isGoal(): #проверка на начальное состояние
            self.solved = True #решение есть
            self.solution_moves = 0 #количество шагов для решения 0
            self.solution = [self.board] #состояние доски окончательное
            return

        if self.heap_type == 'heap':
            main = BinaryHeap()  # Создаем экземпляр класса кучи
            twin = BinaryHeap()  # Создаем экземпляр класса кучи
        else:
            main = SortedArray()  # Создаем экземпляр класса отсортированного массива
            twin = SortedArray()  # Создаем экземпляр класса отсортированного массива

        main.insert(Search(self.board, 0, None, self.priority))
        twin.insert(Search(self.board.twin(), 0, None, self.priority))

        while True: #основной цикл поиска
            if not main.is_empty(): #проверка, пуста ли куча
                current = main.pop() #достаём элемент кучи
                self.search_nodes += 1 #добавление счётчика состояния игры

                if current.board.isGoal(): #проверка на решение в основной куче
                    self.solved = True
                    self.solution_moves = current.moves
                    self.solution = []
                    node = current
                    while node:
                        self.solution.insert(0, node.board)
                        node = node.previous
                    return

                #цикл обрабатывающий соседей для основной кучи
                for neighbor_board in current.board:
                    neighbor_key = self.board_to_key(neighbor_board)
                    if neighbor_key not in self.visited:
                        self.visited[neighbor_key] = True
                        new_node = Search(neighbor_board, current.moves + 1, current, self.priority)
                        main.insert(new_node)

            #проверка очереди на решаемость
            if not twin.is_empty(): #если куча не пустая
                twin_cur = twin.pop()
                if twin_cur.board.isGoal():
                    self.solved = False
                    return

                # цикл обрабатывающий соседей
                for neighbor_board in twin_cur.board:
                    neighbor_key = self.board_to_key(neighbor_board)
                    if neighbor_key not in self.visited:
                        self.visited[neighbor_key] = True
                        new_node = Search(neighbor_board, twin_cur.moves + 1, twin_cur, self.priority)
                        twin.insert(new_node)

    def board_to_key(self, board):#преобразование доски в ключ
        return tuple(tuple(row) for row in board.board)

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
        return str(self.board).replace('[', ' ').replace(']', ' ').replace('0', ' ')

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

def factorial_borders():
    borders = [0]
    total = 0
    for N in range(2, 21):  # считаем факториалы для длин досок от 2 (так как это минимальная длина доски) до 20 (можно увеличить, но числа уже очень большие)
        n = N * N
        total += math.factorial(n)
        borders.append(total)
    return borders


Borders = factorial_borders()


# Функция, которая преобразует код и размерность в доску. Работает по принципу факториального кодирования
def code_and_n_to_board(code, n):

    numbers = list(range(n))  # доступные числа от 0 до n-1
    board = []

    for i in range(n - 1, -1, -1):
        f = math.factorial(i)
        index = code // f
        code %= f
        board.append(numbers.pop(index))  # выбираем index-й элемент
    return board


 # Определяет размерность доски и вызывает функцию code_and_n_to_board
def code_to_board(code):
    for N in range(2, len(Borders)):
        start = Borders[N - 2] + 1  # начальный код для доски размера N×N
        end = Borders[N - 1]  # конечный код для доски размера N×N (включительно)

        if start <= code <= end:
            code = code - start  # факториальный ранг внутри этой размерности
            n = N * N  # длина перестановки
            return code_and_n_to_board(code, n)

    raise ValueError("Слишком большой глобальный номер: превышает поддерживаемый диапазон.")


def analyze_heuristics_with_codes(start_code, end_code): #Анализирует производительность двух эвристик на заданных кодах состояний
    manhattan_moves = []
    hamming_moves = []

    for code in range(start_code, end_code + 1):
        #Пробуем проверить состояние
        try:
            board_numbers = code_to_board(code)
            N = int(math.sqrt(len(board_numbers)))
            board = np.array(board_numbers).reshape((N, N))
            board = Board(board)

            # Тест с Манхэтонновским расстоянием
            solver = Solver(board, 'manhattan', 'heap')

            if solver.isSolvable(): #Проверка на решаемость Манхэттоном
                manhattan_moves.append(solver.moves())
            else:
                manhattan_moves.append(-1)

            # Тест с расстоянием Хэмминга
            solver = Solver(board, 'hamming', 'heap')

            if solver.isSolvable(): #Проверка на решаемость Хээмингом
                hamming_moves.append(solver.moves())
            else:
                hamming_moves.append(-1)

        except:
            print(f"Пропуск кода {code}")
            continue

    return manhattan_moves, hamming_moves

#Построим графики для анализа зависимости количества ходов от состояния игры
def plot_heuristics_comparison(manhattan_moves, hamming_moves, start_code):

    plt.figure(figsize=(15, 10))

    # Фильтруем нерешаемые состояния
    valid_indices = [i for i in range(len(manhattan_moves))
                     if manhattan_moves[i] != -1 and hamming_moves[i] != -1]

    manhattan_valid = [manhattan_moves[i] for i in valid_indices]
    hamming_valid = [hamming_moves[i] for i in valid_indices]
    codes = [i + start_code for i in valid_indices]

    # График количества ходов
    plt.subplot(2, 1, 1)
    plt.plot(codes, manhattan_valid, 'b-', label='Манхэттенское расстояние')
    plt.plot(codes, hamming_valid, 'r-', label='Расстояние Хэмминга')
    plt.xlabel('Код состояния')
    plt.ylabel('Количество ходов')
    plt.title('Сравнение количества ходов')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main_analysis():# основная функция для анализа
    global Borders
    Borders = factorial_borders() #получаем количество состояний для досок

    # Анализируем первые 100 состояний для досок 3x3
    start_code = Borders[1] + 1  # Начало кодов для досок 3x3
    end_code = start_code + 99  # Берем 100 состояний

    print("Анализ эвристик...")
    manhattan_moves, hamming_moves = analyze_heuristics_with_codes(start_code, end_code)

    print("Построение графика...")
    plot_heuristics_comparison(manhattan_moves, hamming_moves, start_code)


def read_board_from_file(filename): #Считываем начальное состояние доски из файла
    try:
        with open(filename, 'r') as f:
            N = int(f.readline().strip()) #Считываем первый параметр. Т.е. Размерность доски
            # Читаем саму доску
            board = []
            for _ in range(N):
                row = list(map(int, f.readline().strip().split()))
                if len(row) != N:
                    raise ValueError(f"Неверное количество чисел в строке. Ожидалось {N}, получено {len(row)}")
                board.append(row)

            # Проверяем, что все числа от 0 до N*N-1 присутствуют
            numbers = set(range(N * N))
            for row in board:
                for num in row:
                    if num not in numbers:
                        raise ValueError(f"Неверное число на доске: {num}")
                    numbers.remove(num)
            return np.array(board)

    except:
        return None


def solve_from_file(filename, priority='manhattan', heap_type='heap'): #Запускаем решение задачи для состояния из файла
    # Читаем доску из файла
    board_array = read_board_from_file(filename)
    if board_array is None:
        return

    # Создаем объект доски
    board = Board(board_array)

    print("Начальное состояние доски:")
    print(board)

    # Создаем решатель и ищем решение
    solver = Solver(board, priority, heap_type)

    if solver.isSolvable():
        print(f"\nРешение найдено за {solver.moves()} ходов:")
        for i, step in enumerate(solver):
            print(f"\nШаг {i}:")
            print(step)
    else:
        print("\nЗадача не имеет решения!")


def compare_solvers(start_code, end_code):#Сравнивает время выполнения двух решателей на одинаковых состояниях
    heap_times = []
    array_times = []

    print("Сравнение решателей...")
    for code in range(start_code, end_code + 1):
        try:
            # Создаем доску из кода
            board_numbers = code_to_board(code)
            N = int(math.sqrt(len(board_numbers)))
            board = np.array(board_numbers).reshape((N, N))
            board = Board(board)

            # Тест с бинарной кучей
            start_time = time.time()
            solver_heap = Solver(board, 'manhattan', 'heap')
            heap_time = time.time() - start_time

            # Тест с отсортированным массивом
            start_time = time.time()
            solver_array = Solver(board, 'manhattan', 'array')
            array_time = time.time() - start_time

            # Проверяем, что оба решателя нашли решение
            if solver_heap.isSolvable() and solver_array.isSolvable():
                heap_times.append(heap_time)
                array_times.append(array_time)

        except:
            continue

    return heap_times, array_times


def analyze_solvers():#Основная функция для анализа решателей
    global Borders
    Borders = factorial_borders()

    # Анализируем первые 50 состояний для досок 3x3
    start_code = Borders[1] + 1  # Начало кодов для досок 3x3
    end_code = start_code + 49  # Берем 50 состояний

    heap_times, array_times = compare_solvers(start_code, end_code)

    print("\nСтатистика:")
    print(f"Бинарная куча:")
    print(f"  Среднее время: {sum(heap_times) / len(heap_times):.4f} сек")
    print(f"  Общее время: {sum(heap_times):.4f} сек")

    print(f"\nОтсортированный массив:")
    print(f"  Среднее время: {sum(array_times) / len(array_times):.4f} сек")
    print(f"  Общее время: {sum(array_times):.4f} сек")


if __name__ == '__main__':
    solve_from_file('board.txt')
    analyze_solvers()
    main_analysis()