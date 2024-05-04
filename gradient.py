import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Define the linear regression model
def linear_regression(x, m, b):
    return m * x + b

# Define the mean squared error (MSE) loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define the gradient of the loss function with respect to parameters
def compute_gradient(x, y, m, b):
    y_pred = linear_regression(x, m, b)
    grad_m = -2 * np.mean(x * (y - y_pred))
    grad_b = -2 * np.mean(y - y_pred)
    return np.array([grad_m, grad_b])

# Gradient Descent optimization algorithm
def gradient_descent(x, y, learning_rate, iterations):
    m = np.random.randn() # Random initialization of slope
    b = np.random.randn() # Random initialization of bias
    losses = []
    for _ in range(iterations):
        gradient = compute_gradient(x, y, m, b)
        m -= learning_rate * gradient[0]
        b -= learning_rate * gradient[1]
        y_pred = linear_regression(x, m, b)
        loss = mean_squared_error(y, y_pred)
        losses.append(loss)
    return m, b, losses

def gradient():
        # Generate sample data
        np.random.seed(0)
        X = np.random.rand(100)
        Y = 2 * X + 1 + np.random.randn(100) * 0.1  # y = 2x + 1 + noise

        # Streamlit UI
        st.title('Gradient Descent Visualization for Linear Regression')
        st.sidebar.header('Parameters')
        learning_rate = st.sidebar.slider('Learning Rate', 0.001, 0.3, 0.1, step=0.001)
        iterations = st.sidebar.slider('Iterations', 10, 100, 10, step=10)

        # Perform gradient descent
        m_opt, b_opt, losses = gradient_descent(X, Y, learning_rate, iterations)

        # Plot the data and the linear regression line
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Data Points'))
        x_range = np.linspace(0, 1, 100)
        y_range = m_opt * x_range + b_opt
        fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Regression Line'))
        fig.update_layout(title='Linear Regression with Gradient Descent',
                        xaxis_title='X',
                        yaxis_title='Y',
                        showlegend=True)

        # Plot the loss curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=np.arange(iterations), y=losses, mode='lines', name='Loss'))
        fig_loss.update_layout(title='Loss Curve',
                            xaxis_title='Iterations',
                            yaxis_title='Mean Squared Error')

        # Display the UI components
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig_loss, use_container_width=True)
        st.sidebar.header('Optimized Parameters')
        st.sidebar.text(f'Slope (m): {m_opt}')
        st.sidebar.text(f'Intercept (b): {b_opt}')
