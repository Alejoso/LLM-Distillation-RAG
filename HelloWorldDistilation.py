import torch 
from torch import nn #Neural Network

# When I use this class (nn) I gotta use a children class which allows me to define my Neural Network

class TeacherModel(nn.Module): 
    # 1. "init" Method, here we define the Architecture of the Neural Network
    def __init__(self): 
        super().__init__() 
        self.network = nn.Sequential(
            nn.Linear (7, 60),
            nn.ReLU(), # Activation Function 
            nn.Linear(60, 2),
        )

    # 2. Here we process the function 
    def forward (self, x): 
        logits = self.network(x)
        return logits

class StudentModel(nn.Module): 

    def __init__(self): 
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear (7, 12), 
            nn.ReLU(),
            nn.Linear(12, 2), 
        )
    
    def forward (self, x): 
        logits = self.network(x)
        return logits


# Dummy data
data = torch.randint(low=0, high=10, size=(1000, 7)).float()

teacherModel = TeacherModel()
studentModel = StudentModel()

# Freeze teacher
for p in teacherModel.parameters():
    p.requires_grad = False # The teacher don't learn because its answers are "true"

# Loss & optimizer (train student)
distill_loss = nn.MSELoss()
optimizer = torch.optim.SGD(studentModel.parameters(), lr=0.01)
teacherLogits = teacherModel(data)

# Distillation loop (Hello World)
for epoch in range(1000):
    
    studentLogits = studentModel(data) 

    loss = distill_loss(studentLogits, teacherLogits) # This is the comparison between logits of both models

    optimizer.zero_grad() # Clean old gradients
    loss.backward() # Here it calculate the neural network weights based on the loss 
    optimizer.step() # Assign new weights

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Distillation Loss: {loss.item():.4f}")

# Parameter count
totalParamsTeacher = sum(p.numel() for p in teacherModel.parameters())
totalParamsStudent = sum(p.numel() for p in studentModel.parameters())

print("Total parameters Teacher:", totalParamsTeacher)
print("Total parameters Student:", totalParamsStudent)




