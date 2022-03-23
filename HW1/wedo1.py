import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing


def main(verbosity=False):
    st.header("Build a Mixture Regression Model")
    st.markdown("""
    In this we do session, we will be implementing a mixture linear regressor:

    1. A general linear regression model
    2. A **building age** level specific model.

    Ultimate loss function will be optimizing the parameters of both models at the same time
    """)

    st.header("Dataset")
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    st.dataframe(X)

    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target, HouseAgeGroup=(X['HouseAge'].values / 10).astype(np.int)))

    st.dataframe(df)

    st.subheader("House Age independent General Model")
    fig = px.scatter(df, x="MedInc", y="Price", trendline="ols")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model per House Age Group")
    group = st.selectbox("House Age Group", [0, 1, 2, 3, 4, 5])
    fig = px.scatter(df[df["HouseAgeGroup"] == group], x="MedInc", y="Price", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

    if verbosity:
        st.subheader("Number of instance by House Age Group")
        st.dataframe(df.groupby('HouseAgeGroup').count())

    st.subheader("Formulating the mixture Group")

    st.markdown("#### General Model")
    st.latex(r"\hat{y}^{0}_i=\beta_0 + \beta_1 x_i")
    st.markdown("#### House Age Group specific Models")
    st.latex(r"\hat{y}^{1}_i=\gamma^{color}_0 + \gamma^{color}_1 x_i")

    st.markdown("#### Loss Function")

    st.write("Final prediction is a combination (mixtures) of two models")
    st.latex(r"\hat{y}_i = p \hat{y}^{0}_i + (1-p) \hat{y}^{1}_i")

    st.latex(
        r"L(\beta_0,\beta_1,\gamma^{color}_{0},\gamma^{color}_{1})=\sum_{i=1}^{N}{(y_i - \hat{y}_i )^2 }")

    st.markdown("#### Partial derivatives")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,\gamma^{color}_{0},\gamma^{color}_{1})}{\partial \beta_0}=-2p\sum_{i=1}^{N}{(y_i - \hat{y}_i) }")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,\gamma^{color}_{0},\gamma^{color}_{1})}{\partial \beta_1}=-2p\sum_{i=1}^{N}{(y_i - \hat{y}_i)x_i }")

    st.write("Note that we calculate each group gradient separately")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,\gamma^{color}_{0},\gamma^{color}_{1})}{\partial \gamma^{color}_{0}}=-2 (1-p )\sum_{i \in HouseAgeGroup = color}{(y_i - \hat{y}_i) }")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,\gamma^{color}_{0},\gamma^{color}_{1})}{\partial \gamma^{color}_{1}}=-2(1-p )\sum_{i \in HouseAgeGroup = color}{(y_i - \hat{y}_i)x_i }")

    st.write(
        "**Mixture Ratio (p)** hyperparameter allows us to choose between pure HouseAgeGroup based model (`p=0`) and a common model over all instances (`p=1`)")

    p = st.slider("Mixture Ration (p)", 0.0, 1.0, value=0.0)
    beta, gamma = reg(df['MedInc'].values, df['Price'].values, df['HouseAgeGroup'].values,
                      p=p,
                      verbose=verbosity)

    st.subheader(f"General Model with p={p:.2f} contribution")
    st.latex(fr"Price = {beta[1]:.4f} \times MedInc + {beta[0]:.4f}")

    st.subheader(f"House Age Group Models with p={1 - p:.2f} contribution")
    st.latex(fr"Price = {gamma[0][1]:.4f} \times MedInc + {gamma[0][0]:.4f}")
    st.latex(fr"Price = {gamma[1][1]:.4f} \times MedInc + {gamma[1][0]:.4f}")
    st.latex(fr"Price = {gamma[2][1]:.4f} \times MedInc + {gamma[2][0]:.4f}")
    st.latex(fr"Price = {gamma[3][1]:.4f} \times MedInc + {gamma[3][0]:.4f}")
    st.latex(fr"Price = {gamma[4][1]:.4f} \times MedInc + {gamma[4][0]:.4f}")


def reg(x, y, group, p=0.3, verbose=False):
    beta = np.random.random(2)
    gamma = dict((k, np.random.random(2)) for k in range(6))

    if verbose:
        st.write(beta)
        st.write(gamma)
        st.write(x)

    alpha = 0.002
    my_bar = st.progress(0.)
    n_max_iter = 100
    for it in range(n_max_iter):

        err = 0
        for _k, _x, _y in zip(group, x, y):
            y_pred = p * (beta[0] + beta[1] * _x) + (1 - p) * (gamma[_k][0] + gamma[_k][1] * _x)

            g_b0 = -2 * p * (_y - y_pred)
            g_b1 = -2 * p * ((_y - y_pred) * _x)

            # st.write(f"Gradient of beta0: {g_b0}")

            g_g0 = -2 * (1 - p) * (_y - y_pred)
            g_g1 = -2 * (1 - p) * ((_y - y_pred) * _x)

            beta[0] = beta[0] - alpha * g_b0
            beta[1] = beta[1] - alpha * g_b1

            gamma[_k][0] = gamma[_k][0] - alpha * g_g0
            gamma[_k][1] = gamma[_k][1] - alpha * g_g1

            err += (_y - y_pred) ** 2

        print(f"{it} - Beta: {beta}, Gamma: {gamma}, Error: {err}")
        my_bar.progress(it / n_max_iter)

    return beta, gamma


if __name__ == '__main__':
    main(st.sidebar.checkbox("verbosity"))