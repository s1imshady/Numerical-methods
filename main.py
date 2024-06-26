import numpy as np
import matplotlib.pyplot as plt

#массив который будет содержать имена методов
methods=['relaxation','min_unbound','rapid_descent','minimum_correction','conjugate_gradient ']


def make_system(n):
    A = np.random.randint(0,25 ,(n,n))
    A = np.dot(A, A.T)
    b=np.random.randint(1,25,(1,n))
    return A,b


asnwer = []
time = []
accuracy = []
matrix_array=[]
b_array=[]

for n in range(5,30,5):
    A,b=make_system(n)
    matrix_array.append(A)
    b_array.append(b)

#заданная точность
eps=10**(-5)
#получение результатов
for A,b in  (matrix_array, b_array):
    result1 = func1(A,b,eps)
    result2 = func2(A,b,eps)
    result3 = func3(A,b,eps)
    result4 = func4(A,b,eps)
    result5 = func5(A,b,eps)
    asnwer.append([result1[0],result2[0],result3[0],result4[0],result5[0]])
    time.append([result1[1],result2[1],result3[1],result4[1],result5[1]])
    accuracy.append([result1[2],result2[2],result3[2],result4[2],result5[2]])

#преобразуем к массивам numpy для удобства
answer = np.array(asnwer)
accuracy = np.array(accuracy)
time = np.array(time)

#создаем графики
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

#графики времени выполнения
for i in range(5):
    axs[0].plot(methods, time[:, i], label=f'{(i+1)*5}')
axs[0].set_title('Time')
axs[0].set_xlabel('Method')
axs[0].set_ylabel('time')
axs[0].legend()

#графики точности
for i in range(5):
    axs[1].plot(methods, accuracy[:, i],label=f'{(i+1)*5}')
axs[1].set_title('Accuracy')
axs[1].set_xlabel('Method')
axs[1].set_ylabel('acc')
axs[1].legend()



plt.tight_layout()
plt.show()
