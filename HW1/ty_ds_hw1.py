import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
import plotly.express as px
import plotly.graph_objects as go
import wedo1
from sklearn.metrics import mean_squared_error

def convexity_proof(x,y,th,alpha=0.000001):
    loss_all = []
    for beta in np.linspace(-1, 1, 100):
        diff = (y - beta * x)
        loss = np.power(diff, 2)
        loss[loss > np.power(th,2)] = np.power(th,2)
        cost = np.sum(loss)
        loss_all.append(cost)
    df_loss = pd.DataFrame(dict(beta1=np.linspace(-1, 1, 100), loss_all=loss_all))

    fig = px.scatter(df_loss, x="beta1", y="loss_all")
    st.subheader("Loss function graph for beta values between [-1,1]")
    st.plotly_chart(fig, use_container_width=True)

def ls(x, y, th,alpha=0.000001) -> np.ndarray:

    beta = np.random.random(2)

    loss_all = [] #Cost
    betas = []
    print("starting sgd")
    print("started")
    for i in range(100):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        loss = np.power(y - y_pred, 2) # Calculating the loss
        loss[loss > np.power(th, 2)] = np.power(th, 2)
        loss_all.append(np.sum(loss))


        g_b0 = -2 * (y - y_pred).sum()
        g_b1 = -2 * (x * (y - y_pred)).sum()

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1
        betas.append(beta[1])

        if np.linalg.norm(beta - beta_prev) < 0.001:
            print(f"I do early stoping at iteration {i}")
            break

    df_loss = pd.DataFrame(dict(betas_ls=betas, loss_all_ls=loss_all))

    fig = px.scatter(df_loss, x="betas_ls", y="loss_all_ls")

    st.plotly_chart(fig, use_container_width=True)

    return beta


def ls_l2(x, y, lam,th, alpha=0.000001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)
    loss_all_l2 = []  # Cost
    betas_l2 = []
    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        loss = np.power(y - y_pred, 2)  # Calculating the loss
        loss[loss > np.power(th, 2)] = np.power(th, 2)
        loss_all_l2.append(np.sum(loss))

        g_b0 = -2 * (y - y_pred).sum() + 2 * lam * beta[0]
        g_b1 = -2 * (x * (y - y_pred)).sum() + 2 * lam * beta[1]

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1
        betas_l2.append(beta[1])


        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break

    df_loss_l2 = pd.DataFrame(dict(betas_l2=betas_l2, loss_all_l2=loss_all_l2))

    fig = px.scatter(df_loss_l2, x="betas_l2", y="loss_all_l2")

    st.plotly_chart(fig, use_container_width=True)

    return beta


def main():
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target
    df = pd.DataFrame(dict(MedInc=X['MedInc'], Price=cal_housing.target))
    st.subheader("Dataframe")
    st.dataframe(df)

    lam1 = st.slider("Regularization Multiplier for L2 (lambda)", 0.001, 10., value=0.3)
    th = st.slider("Threshold", 0.001, 10., value=1.5)

    convexity_proof(df["MedInc"], y, th) #showing convexity of Loss Function

    st.write("**Graphs might need more zoom to show convexity**")

    beta_ls = ls(df["MedInc"], y, th)
    st.write("Ls Beta values:")
    st.latex(fr"\beta_0={beta_ls[0]:.4f}, \beta_1={beta_ls[1]:.4f}")

    beta_ls_l2 = ls_l2(df["MedInc"], y, lam1, th)
    st.write("Ls_l2 Beta values:")
    st.latex(fr"\beta_0={beta_ls_l2[0]:.4f}, \beta_1={beta_ls_l2[1]:.4f}")

    p = st.slider("Mixture Ration (p)", 0.0, 1.0, value=0.0)
    beta_wedo, gamma = wedo1.reg(df['MedInc'].values, df['Price'].values,(X['HouseAge'].values / 10).astype(np.int),p=p)

    y_pred_ls = beta_ls[0] + beta_ls[1] * df["MedInc"]
    y_pred_ls_l2 = beta_ls_l2[0] + beta_ls_l2[1] * df["MedInc"]
    y_pred_wedo = beta_wedo[0] + beta_wedo[1] * df["MedInc"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["MedInc"], y=y, mode='markers', name='data points'))
    fig.add_trace(go.Scatter(x=df["MedInc"], y= y_pred_ls, mode='lines', name='least square'))

    fig.add_trace(go.Scatter(x=df["MedInc"], y= y_pred_ls_l2, mode='lines', name='regression + L2'))

    fig.add_trace(go.Scatter(x=df["MedInc"], y=y_pred_wedo, mode='lines', name='regression wedo'))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("MSE Results:")
    st.write("Ls MSE: ",mean_squared_error(y,y_pred_ls))
    st.write("Ls L2 MSE: ",mean_squared_error(y, y_pred_ls_l2))
    st.write("Ls Wedo: ",mean_squared_error(y, y_pred_wedo))


if __name__ == '__main__':
    main()