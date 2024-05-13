import numpy as np

def hypothesis(theta0, theta1, x):
    return theta0 + theta1 * x

def cost_function(theta0, theta1, x, y):
    m = len(y)
    h = hypothesis(theta0, theta1, x)
    return (1 / (2 * m)) * np.sum((h - y)**2)

def gradient_descent(theta0, theta1, x, y, learning_rate=0.01, iterations=1000, convergence_threshold=1e-5):
    m = len(y)
    for iteration in range(iterations):
        h = hypothesis(theta0, theta1, x)
        delta_theta0 = (1 / m) * np.sum(h - y)
        delta_theta1 = (1 / m) * np.sum((h - y) * x)
        
        theta0 -= learning_rate * delta_theta0
        theta1 -= learning_rate * delta_theta1

        # Check for convergence based on gradients
        if np.abs(delta_theta0) < convergence_threshold and np.abs(delta_theta1) < convergence_threshold:
            print(f"Converged after {iteration} iterations.")
            break

        # Print cost for every 100 iterations
        if iteration % 100 == 0:
            cost = cost_function(theta0, theta1, x, y)
            print(f"Iteration {iteration}, Cost: {cost}")

    return theta0, theta1

# Example usage:
# Assuming you have data in the form of arrays x and y
# where x is the input features and y is the output
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Initialize theta0 and theta1
theta0 = 0
theta1 = 0

# Run gradient descent
final_theta0, final_theta1 = gradient_descent(theta0, theta1, x, y)

# Print the final parameters
print(f"Final Parameters: theta0 = {final_theta0}, theta1 = {final_theta1}")
