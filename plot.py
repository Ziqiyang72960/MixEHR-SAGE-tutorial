import matplotlib.pyplot as plt

# Given values
values = [142029511.00079358, 142051201.78663895, 142053507.72457275, 142053465.56598538, 142053810.31197143, 142053257.08855778, 142052516.2690454, 142053365.71651772, 142054221.27295595, 142053893.0023763]
step = 39
print(len(values))
def average_every_step(lst):
    return [sum(lst[i:i+step])/step for i in range(0, len(lst), step) if lst[i:i+step]]
averages = average_every_step(values)
print(averages)
epochs = list(range(1, len(averages) + 1))
# epochs = list(range(0, len(averages) + 1))
plt.figure(figsize=(8, 5))
plt.plot(epochs, averages, marker='o')
plt.title('neg-ELBO per Epoch (with rr)')
plt.xlabel('Epoch')
plt.ylabel('neg-ELBO')
plt.grid(True)
plt.show()
