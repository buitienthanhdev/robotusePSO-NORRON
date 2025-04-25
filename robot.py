import pygame
import math
import torch
import numpy as np
from collections import deque


########################################
# PART 1: Xây Dựng Môi Trường Và Robot
########################################
class Envir:
    def __init__(self, dimentions, bg_img):
        pygame.display.set_caption("Omniwheel Robot")
        self.map = pygame.display.set_mode(dimentions)
        self.bg = pygame.image.load(bg_img)
        
        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)


        # Cảm biến
        self.sensor_data = [0,0,0,0,0,0]  
        self.points = []

    def draw(self):
        self.map.blit(self.bg, (0, 0))

    def robot_frame(self, pos, rotation):
        """ Vẽ hệ trục tọa độ gắn với robot """
        n = 35
        centerx, centery = pos
        x_axis = (centerx + n * math.cos(rotation), centery + n * math.sin(rotation))
        y_axis = (centerx + n * math.cos(rotation + math.pi / 2), centery + n * math.sin(rotation + math.pi / 2))

        pygame.draw.line(self.map, self.blue, (float(pos[0]), float(pos[1])), (float(x_axis[0]), float(x_axis[1])), 3)
        pygame.draw.line(self.map, self.green, (float(pos[0]), float(pos[1])), (float(y_axis[0]), float(y_axis[1])), 3)


    def robot_sensor(self, pos, points):
        for point in points:
            pygame.draw.line(self.map, (0, 255, 0), (int(pos[0]), int(pos[1])), (int(point[0]), int(point[1])))
            pygame.draw.circle(self.map, (0, 255, 0), (int(point[0]), int(point[1])), 5)


class Robot:
    def __init__(self, startpos, robotImg):
        self.x, self.y = torch.tensor(startpos, dtype=torch.float32)
        self.theta = torch.tensor(0.0, dtype=torch.float32)
        self.img = pygame.image.load(robotImg)
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(float(self.x), float(self.y)))

        # Kích thước theo pixel
        self.r = torch.tensor(9.0, dtype=torch.float32)  # Bán kính bánh xe
        self.l = torch.tensor(24.0, dtype=torch.float32) # Bán kính xe

        # Vận tốc bánh xe
        self.v1 = self.v2 = self.v3 = torch.tensor(0.0, dtype=torch.float32)

        # Động học thuận
        self.vx = self.vy = self.omega = torch.tensor(0.0, dtype=torch.float32)

        # Cảm biến 
        self.sensor_data = []
        self.points = []

        # thuộc tính động học thuận

        # Chuyển đổi góc sang Tensor
        pi_4 = torch.tensor(math.pi / 4)
        pi_11_12 = torch.tensor(11 * math.pi / 12)
        pi_19_12 = torch.tensor(19 * math.pi / 12)

        self.J1f = torch.tensor([
            [torch.sin(pi_4), -torch.cos(pi_4), -self.l * torch.cos(pi_4)],
            [torch.sin(pi_11_12), -torch.cos(pi_11_12), -self.l * torch.cos(pi_4)],
            [torch.sin(pi_19_12), -torch.cos(pi_19_12), -self.l * torch.cos(pi_4)]
        ], dtype=torch.float32)

        self.J1f_inv = torch.linalg.inv(self.J1f)

        self.J2 = torch.diag(torch.tensor([self.r, self.r, self.r], dtype=torch.float32))

    def move(self, wheel_speeds, dt):
        self.v1 = wheel_speeds[0, 0].item()
        self.v2 = wheel_speeds[1, 0].item()
        self.v3 = wheel_speeds[2, 0].item()


        R_theta = torch.tensor([
            [torch.cos(self.theta), torch.sin(self.theta), 0],
            [-torch.sin(self.theta), torch.cos(self.theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        

        V_banh = torch.tensor([[self.v1], [self.v2], [self.v3]], dtype=torch.float32)

        # Tính toán động học thuận
        V_xe = torch.linalg.inv(R_theta) @ self.J1f_inv @ self.J2 @ V_banh

        self.vx, self.vy, self.omega = V_xe.flatten()

        self.x += self.vx*dt
        self.y += self.vy*dt
        self.theta = (self.theta + self.omega*dt) % (2 * math.pi)

        # Cập nhật hình ảnh robot
        self.rotated = pygame.transform.rotozoom(self.img, -math.degrees(self.theta.item()), 1)
        self.rect = self.rotated.get_rect(center=(float(self.x), float(self.y)))

    def draw(self, map):
        map.blit(self.rotated, self.rect)


    def update_sensor(self, pixel_array):
        angles = [self.theta + i * (2 * math.pi / 12) for i in range(12)]
        edge_points = []
        edge_distance = []
        for angle in angles:
            distance = 0
            edge_x, edge_y = (int(self.x), int(self.y)) 
            while distance < 100 and (0 <= edge_x < 855) and (0 <= edge_y < 1000):
                if np.all(pixel_array[int(edge_x), int(edge_y)] == (0, 0, 0)):  # Kiểm tra màu đen
                    break
                edge_x = int(self.x + distance*math.cos(angle))
                edge_y = int(self.y + distance*math.sin(angle))
                distance += 2
            edge_points.append((edge_x, edge_y))
            edge_distance.append(distance)
        self.sensor_data = edge_distance
        self.points = edge_points


########################################
# PART 2: Xây Dựng mạng Nơron
########################################

def decode_particle(params):
    """
    Giải mã vector tham số (2367 phần tử) thành các trọng số và bias:
      - FC1: 16x48 + 48 Nhận cảm giác – Phân tích tình huống
      - FC2: 48x24 + 24 Chiến thuật – Xác định xu hướng hành vi
      - FC3: 24x12 + 12 Chuẩn bị quyết định
      - Output: 12x3 + 3 Quyết định
    """
    idx = 0

    # FC1
    W1 = params[idx:idx + 16*48].reshape(16, 48)
    idx += 16*48
    b1 = params[idx:idx + 48].reshape(48, 1)
    idx += 48

    # FC2
    W2 = params[idx:idx + 48*24].reshape(48, 24)
    idx += 48*24
    b2 = params[idx:idx + 24].reshape(24, 1)
    idx += 24

    # FC3
    W3 = params[idx:idx + 24*12].reshape(24, 12)
    idx += 24*12
    b3 = params[idx:idx + 12].reshape(12, 1)
    idx += 12

    # Output
    W4 = params[idx:idx + 12*3].reshape(12, 3)
    idx += 12*3
    b4 = params[idx:idx + 3].reshape(3, 1)

    return W1, b1, W2, b2, W3, b3, W4, b4


def neural_net(X, params, speed_factor=5):
    """
    Mạng nơron:
    - Input: 16x1 (12 cảm biến + sinθ, cosθ + x, y)
    - 3 hidden layers (ReLU)
    - Output: 3x1 (tanh) điều khiển tốc độ 3 bánh
    """
    W1, b1, W2, b2, W3, b3, W4, b4 = decode_particle(params)

    h1 = torch.relu(torch.matmul(W1.T, X) + b1)
    h2 = torch.relu(torch.matmul(W2.T, h1) + b2)
    h3 = torch.tanh(torch.matmul(W3.T, h2) + b3)
    output = torch.tanh(torch.matmul(W4.T, h3) + b4) * speed_factor

    return output  # 3x1


########################################
# PART 3: Xây Dựng Hàm Mục Tiêu
########################################

def evaluate_population(pop, env, pixel_array, max_time, dt=0.1, render=True ):

    """
    Mô phỏng đồng thời toàn bộ robot trong quần thể với các bộ tham số trong pop.
    Trả về danh sách fitness của từng cá thể.
    Vì khi huấn luyện thời gian mỗi vòng lặp sẽ xử lý rất lâu tầm 1 giây hoặc hơn, còn khi đã có mô hình và mô phỏng một xe thì thời gian xử lý lại rất nhanh 
    (khoảng 0.01 giây) nên ta sẽ đặt dt cố định là 0,1 giây là đủ cho mỗi lần robot xử lý việc lấy dữ liệu và điều khiển, như vậy sẽ giúp robot chạy hiệu quả 
    hơn và còn tối ưu tài nguyên tính toán, giúp quá trình huấn luyện tốt hơn, do đó, khi huấn luyện em sẽ đặt dt cố định là 0.1 cho mỗi vòng lặp, khi mô phỏng thì 
    sẽ dùng thời gian thực và kiếm soát dt, khi bộ đếm đủ 0.1 giây thì bắt đầu vòng lặp điều khiển tiếp theo.
    """
    pop_size = pop.shape[0] # Số lượng cá thể trong quần thể (lấy từ số hàng của tensor pop).

    # Tạo danh sách robot cho toàn bộ cá thể, khởi tạo tại vị trí ban đầu
    robots = [Robot((385, 30),'xe.png') for _ in range(pop_size)]

    # Các biến tính toán fitness_values
    base = 3000
    time_coeff = -1             # Thưởng thời gian       
    collision_penalty = 500     # Phạt va chạm
    reward_coeff = -300         # Thưởng khi đi qua điểm chiến lược
    stuck_bonus = 500           # Phạt va chạm
    goal_reward = -1000         # Thưởng đến đích
    stop_reward = -1000         # Thưởng dừng khi đến đích
    else_bonus = 500          
    
    
    # Các biến lưu trạng thái mô phỏng
    fitness_values = [None] * pop_size
    hoanthanh = [False] * pop_size
    in_goal = [False] * pop_size # Đánh dấu robot đã đến đích



    # Tính toán điểm chiến lược
    reward_points = [(386, 160), (521, 366), (414, 586), (305, 678), (143, 560), (196, 867)] # những điểm chiến lược mà robot nên đi qua
    rp_tensor = torch.tensor(reward_points, dtype=torch.float32)
    reward_matrices = [rp_tensor.clone() for _ in range(pop_size)] # ma trận điểm thưởng chưa đi qua của từng robot
    reward_total =  [0] * pop_size # Mỗi điểm chiến lược đi qua +1 điểm thưởng



    goal_x = 467
    goal_y = 970
    vitri = [[] for _ in range(pop_size)]
    present_time = 0
    step = 0 

    
    # Vòng lặp mô phỏng
    while present_time <= max_time and not all(hoanthanh):
            
        if render:
            env.draw()  # Vẽ nền
            
        present_time += dt
        step += 1
        
        # vòng lặp mỗi robot
        for i in range(pop_size):
            if hoanthanh[i]:
                continue

            robot = robots[i]

            # Cập nhật cảm biến
            robot.update_sensor(pixel_array)
            
            # Chuẩn hóa dữ liệu đầu vào cho mạng nơron [-1;1]
            normalized_sensor = [s / 100 for s in robot.sensor_data]
            normalized_x = robot.x / 855
            normalized_y = robot.y / 1000
            normalized_sin = (math.sin(robot.theta) + 1) / 2
            normalized_cos = (math.cos(robot.theta) + 1) / 2

            sensor_tensor = torch.tensor([normalized_x, normalized_y, normalized_sin, normalized_cos] + normalized_sensor ,
                                         dtype=torch.float32).reshape(-1, 1)
            
            # Lấy bộ tham số của cá thể i (đảm bảo đúng kích thước)
            candidate_params = pop[i].reshape(-1)
            # Dự đoán tốc độ bánh xe
            wheel_speeds = neural_net(sensor_tensor, candidate_params)
            
            # Điều khiển robot
            robot.move(wheel_speeds, dt)

            # Lưu lịch sử vị trí
            vitri[i].append((robot.x.item(), robot.y.item()))


            # Tính khoảng cách Euclidean đến đích sau bước di chuyển
            current_distance = math.sqrt((robot.x - goal_x) ** 2 + (robot.y - goal_y) ** 2)
            # Kiểm tra nếu robot đã đến đích
            if current_distance < 25:
                # Đánh dấu rằng robot đã vào vùng đích
                in_goal[i] = True

            # Tính khoảng cách từ robot đến toàn bộ điểm chiến lược còn lại
            if reward_matrices[i].shape[0] > 0:
                # Lấy vị trí hiện tại của robot dưới dạng tensor (1, 2)
                current_pos = torch.tensor([[robot.x, robot.y]], dtype=torch.float32)

                # Tính khoảng cách Euclidean đến từng điểm chiến lược còn lại
                diff = reward_matrices[i] - current_pos  # shape: (num_points, 2)
                dists = torch.norm(diff, dim=1)                # shape: (num_points,)

                # Lọc ra các điểm chưa đi qua 
                mask = dists >= 25
                passed = (~mask).sum().item()  # số điểm vừa "đi qua"
                reward_total[i] += passed      # cộng reward

                # Giữ lại các điểm chưa đi qua
                reward_matrices[i] = reward_matrices[i][mask]



            # Kiểm tra va chạm hoặc ra ngoài biên
            if any(s < 5 for s in robot.sensor_data) or not (0 < robot.x < 855) or not (0 < robot.y < 1000):
                hoanthanh[i] = True
       
                fitness_values[i] = (base + 
                                     time_coeff * step + 
                                     collision_penalty +
                                     reward_coeff * reward_total[i]  )
                continue



            # Kiểm tra 70 vị trí gần nhất để cập nhật betac:
            if (not in_goal[i]) and (step >= 40 ):
                # Chuyển vitri thành tensor, có shape (num_points, 2)
                vitri_tensor = torch.tensor(vitri[i], dtype=torch.float32)
                # Lấy 20 điểm mới nhất
                recent_points = vitri_tensor[-10:]
                # Lấy tối đa 100 điểm trước 20 điểm gần nhất để tối ưu tính toán
                old_points = vitri_tensor[max(0, step - 150):-10]
                # broadcasting để tính hiệu số cho tất cả cặp điểm.
                diff = recent_points.unsqueeze(1) - old_points.unsqueeze(0)  
                #khoảng cách Euclidean
                dists = torch.norm(diff, dim=2)  

                min_dists = dists.min(dim=1)[0]  # Tìm khoảng cách nhỏ nhất của mỗi điểm mới đến tất cả điểm cũ

                # Nếu mọi điểm mới đều nằm trong vi phạm thì robot đang bế tắc
                if (min_dists < 30).all():  
                    hoanthanh[i] = True

                    fitness_values[i] = (base + 
                                        time_coeff * step +
                                        reward_coeff * reward_total[i]+
                                        stuck_bonus )
                    
                    continue



            if in_goal[i] == True:
                if current_distance > 25: # robot đã đến đích nhưng lại đi ra ngoài
                    hoanthanh[i] = True
                    fitness_values[i] =(base+
                                        time_coeff * step +
                                        reward_coeff * reward_total[i] +
                                        goal_reward )
                    
                    continue


            #if  step % 5 == 0: 
            robot.draw(env.map)
            env.robot_frame([robot.x, robot.y], robot.theta)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        pygame.display.update()


    for i in range(pop_size):
        if not hoanthanh[i]:
            if in_goal[i]:
                fitness_values[i] =(base+
                                    time_coeff * step +
                                    reward_coeff * reward_total[i] +
                                    goal_reward +
                                    stop_reward)
                
            else:
                fitness_values[i] =(base+
                                    time_coeff * step +
                                    reward_coeff * reward_total[i] +
                                    else_bonus )
        

        

    return fitness_values


########################################
# PART 4: Huấn Luyện Mô Hình
########################################


# Khởi tạo pygame
pygame.init()

# Khởi tạo môi trường và robot
dims = (855, 1000)
env = Envir(dims, 'map.png')
#robot = Robot((451, 30), 'C:/Users/assas/Desktop/MobileRobot/robot.png')

map_copy = env.bg.copy()
map_copy = map_copy.convert(24)  # Chuyển sang 24-bit RGB
pixel_array = pygame.surfarray.pixels3d(map_copy)

# Các tham số của thuật toán PSO
pop_size = 1000        # Số lượng cá thể (particles)
npar = 2367            # Kích thước vector tham số của mạng
w = 0.5               # Hệ số quán tính
c1 = 1.5              # Hệ số học cá nhân (pbest)
c2 = 1.5              # Hệ số học xã hội (gbest)
max_iteration = 100   # Số vòng lặp PSO

# Lần train sau
pop = torch.load('C:/Users/Thanh/Desktop/AAAAAAAAAAAAAAAAA/Pbest_positionoke3.pt')
#pop = (torch.rand((pop_size, npar))*2)-1
V = torch.zeros_like(pop)
Pbest_position = pop.clone()
Pbest_fitness = torch.ones(pop_size) * 10000000
Gbest_fitness = 10000000
Gbest_position = None


# Các thông số của hàm fitness
max_time = 300  #(giây) thời gian tối đa mỗi robot
# Mảng lưu lịch sử fitness tốt nhất qua các vòng lặp
fitness_history = []


    
# Vòng lặp PSO
for iteration in range(max_iteration):
    
    fitness_values = evaluate_population(pop, env, pixel_array, max_time)
    
    # Cập nhật Pbest và Gbest dựa trên fitness của từng cá thể
    for i in range(pop_size):
        if fitness_values[i] < Pbest_fitness[i]:
            Pbest_fitness[i] = fitness_values[i]
            Pbest_position[i] = pop[i].clone()
            if fitness_values[i] < Gbest_fitness:
                Gbest_fitness = fitness_values[i]
                Gbest_position = pop[i].clone()

    # Lấy mẫu từ top 20% tốt nhất
    elite_num =  pop_size // 10
    elite_idx = torch.topk(torch.tensor(fitness_values), k=elite_num, largest=False).indices

    # Tái sinh cá thể yếu kém nếu cách biệt > 200 điểm so với Gbest
    for i in range(pop_size):
        if fitness_values[i] - Gbest_fitness > 300:

            # Chọn ngẫu nhiên một cá thể từ top tốt nhất để làm "hạt giống" cho cá thể yếu được tái sinh.
            elite_sample = pop[elite_idx[torch.randint(0, len(elite_idx), (1,))]].clone() # sao chép ra hẳn 1 bản, tránh liên kết tham chiếu trong Tensor.
            # Tạo nhiễu Gaussian ngẫu nhiên cùng kích thước với elite_sample.
            noise = torch.randn_like(elite_sample) * 0.2
            pop[i] = elite_sample + noise
            V[i] = torch.zeros_like(V[i])  # Reset vận tốc về 0

    # Thêm GA để tăng khả năng khám phá lời giải, ý tưởng là sau một số vòng lặp cố định, lai ghép một phần cá thể tốt nhất để tạo đột phá
    if iteration % 20 == 0:  # Mỗi 20 vòng lặp
        print("Lai ghép GA")

        elites = [pop[i] for i in elite_idx] # Tạo danh sách elites chứa các cá thể mạnh nhất để làm bố mẹ trong quá trình lai ghép.

        num_offspring = max(1, pop_size // 10)  # Linh hoạt theo kích thước quần thể, thay thế 20% cá thể tệ nhất

        for _ in range(num_offspring):
            # Chọn 2 bố mẹ ngẫu nhiên từ elite
            p1 = elites[torch.randint(0, elite_num, (1,)).item()]
            p2 = elites[torch.randint(0, elite_num, (1,)).item()]

            # Lai ghép pha trộn
            alpha = torch.rand(npar)  # Tỷ lệ trộn ngẫu nhiên cho mỗi tham số
            child = alpha * p1 + (1 - alpha) * p2  # Lai ghép 
    
            # Đột biến nhỏ xác suất 30%
            if torch.rand(1).item() < 0.3:
                mut_pos = torch.randint(0, npar, (1,)) # Chọn 1 gene ngẫu nhiên
                child[mut_pos] += (torch.randn(1) * 0.1) # Thêm nhiễu Gaussian nhỏ (±0.1) để đột biến.

            # Thay thế cá thể tệ nhất hiện tại
            worst_idx = torch.argmax(torch.tensor(fitness_values)).item()
            pop[worst_idx] = child
            V[worst_idx] = torch.zeros_like(V[worst_idx]) # reset vận tốc về 0


        
    # Cập nhật vận tốc và vị trí các cá thể
    r1 = torch.rand((pop_size, npar))
    r2 = torch.rand((pop_size, npar))
    
    # Công thức cập nhật vận tốc PSO
    V = w * V + c1 * r1 * (Pbest_position - pop) + c2 * r2 * (Gbest_position - pop)
    
    # Giới hạn vận tốc để tránh phân kỳ
    #V = torch.clamp(V, -1.0, 1.0)
    
    
    # Cập nhật vị trí các cá thể
    pop = pop + V

    # Thêm cơ chế đột biến
    mutation_rate = 0.1 * (1 - iteration / max_iteration)  # Giảm dần từ 0.1 xuống 0
    mutation_mask = (torch.rand((pop_size, npar)) < mutation_rate).float()
    mutation_values = (torch.rand((pop_size, npar)) * 2 - 1) * 0.1  # Đột biến trong khoảng [-0.1, 0.1]
    pop = pop + mutation_mask * mutation_values

    # Giới hạn giá trị tham số trong khoảng [-1, 1]
    #pop = torch.clamp(pop, -1.0, 1.0)
    
    # Điều chỉnh hệ số  giảm dần theo vòng lặp (giả sử giảm tuyến tính)
    w = max(0.2, 0.5 - (0.3 * iteration / max_iteration))
    c1 = max(0.5, 1.5 - (1 * iteration / max_iteration))
    c2 = max(0.5, 1.5 - (1 * iteration / max_iteration))
    
    # In thông tin tiến trình mỗi vòng lặp
    print(f"Vòng lặp {iteration} Gbest_fitness : {Gbest_fitness}  ;  Best_fitness: {min(fitness_values)}")

    
    pygame.display.update()


# Sắp xếp Pbest_position theo thứ tự fitness từ nhỏ đến lớn.
# Giả sử fitness càng nhỏ càng tốt.
sorted_indices = torch.argsort(Pbest_fitness, descending=False)  # descending=False để sắp xếp tăng dần
Pbest_position_sorted = Pbest_position[sorted_indices]
torch.save(Gbest_position, "C:/Users/Thanh/Desktop/AAAAAAAAAAAAAAAAA/best_modeloke4.pt")
torch.save(Pbest_position_sorted, "C:/Users/Thanh/Desktop/AAAAAAAAAAAAAAAAA/Pbest_positionoke4.pt")