# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="-s-r8b183fe8"
# # 村田安雄『動的経済システムの最適制御』第2章のシミュレーション
#
# (Version: 0.0.1)
#
# 「第2章 投資の加速度原理の経済での安定化政策」について本にシミュレーション結果があるが、それを「追試」してみる。
#
# ここでの streamplot などの知識は、南裕樹『Python による制御工学入門』から得た。

# + id="SaByRl0m7_GS"
# %matplotlib inline

# + colab={"base_uri": "https://localhost:8080/"} id="jiU2987p_niD" outputId="7147e9ff-62db-49de-a33e-06d3eb487a85"
from time import gmtime, strftime
print("Time-stamp: <%s>" % strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()))

# + id="QcJnm2iN4TmJ"
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np

# + [markdown] id="50Ih5r_a4lwh"
# どれぐらいの範囲の図にすればいいのか本からはわからない。とりあえず、-2.0 から 2.0 を 100 マスで区切ったものにする。

# + id="Uk8cSim84jwK"
width = 2.0

Y, G = np.mgrid[-width:width:100j, -width:width:100j]

# + [markdown] id="nNcMkVaK49dJ"
# この部分、なぜか次のようにすると、streamplot が ValueError: setting an array element with a sequence. …というエラーを吐く。

# + id="ChfZX1xY4lIA"
# G, Y = np.mgrid[-width:width:100j, -width:width:100j]

# + [markdown] id="RG9Cvx7p5Q4y"
# 各種パラメータは本の例のようにとった。

# + id="aViOVXVV5UtB"
s = 0.3
nu = 4
rho = 0.1
phi = 2

w = rho / (1 - nu * rho)
theta = np.sqrt(w ** 2 + phi ** -1)

# + [markdown] id="dxP6v1NE6OKi"
# 次の A は式 (22) そのまま。

# + colab={"base_uri": "https://localhost:8080/"} id="-tEKVyiP5ZNQ" outputId="787690db-0d72-4ab5-9023-3dc4b18e96cd"
A = np.array([[- s * w, w], [w / phi, s * w]])
ev, evec = np.linalg.eig(A)

print(ev)

# + id="2I6ZuDAX5cIK"
dY = A[0, 0] * Y + A[0, 1] * G
dG = A[1, 0] * Y + A[1, 1] * G

# + id="FEJ5fgxx5jwa"
t = np.arange(-width, width, width/200)

# + colab={"base_uri": "https://localhost:8080/", "height": 285} id="_hRUp0v25nii" outputId="0d78c996-9be3-4d55-da9e-724c0ed52425"
fig, ax = plt.subplots()

ax.streamplot(G, Y, dG, dY, density=0.7, color='k')

if ev.imag[0] == 0 and ev.imag[1] == 0:
    ax.plot(t, (evec[1,0]/evec[0,0]) * t, ls='-')
    ax.plot(t, (evec[1,1]/evec[0,1]) * t, ls='-')

ax.set_xlim([-width, width])
ax.set_ylim([-width, width])
ax.set_xlabel('$g$')
ax.set_ylabel('$y$')
None

# + [markdown] id="CopkMxWj5wN6"
# さてここからが今回私がやりたかったことである。「最適化」を解析的に導出するのではなく、力技で scipy.optimize.minimize を使って求めたらどうなるか？…というのを試してみたかった。それをやってみる。

# + id="FEECDVY_5reS"
dt = 0.1

def calc_ys (y0, gs) :
    ys = [y0]
    g0 = gs[0]
    gs = gs[1:]
    y = y0
    g = g0
    for i, ng in enumerate(gs):
        dy = w * g - s * w * y # これは式 (12) そのものである。
        y = y + dy * dt
        g = ng
        ys.append(y)
    # assert len(ys) == len(gs) + 1
    return ys

def calc_score (gs, y0):
    ys = np.array(calc_ys(y0, gs))
    gs = np.array(gs)
    return np.sum(ys ** 2 + phi * gs ** 2)


# + [markdown] id="nikdV3Nz6gvi"
# フィリップスの最適安定化モデルでは、式 (13) の $J = \frac{1}{2}\int^T_0 (y^2 + \phi g ^2) dt$ を目的関数にしているが、ここでは np.sum(ys ** 2 + phi * gs ** 2) を使っている。
#
# 初期値は仮に y0 = -1 にしている。

# + id="CG-TUcT-7Zxp"
y0 = -1

# + id="H5ISiV8y7htS"
res = scipy.optimize.minimize(calc_score, [0] * 100, args=(y0,), method='Nelder-Mead')

# + colab={"base_uri": "https://localhost:8080/", "height": 285} id="WwNhgZMI8QF_" outputId="562c6356-85c9-4006-e145-a79585b0d4a1"
fig, ax = plt.subplots()

ax.streamplot(G, Y, dG, dY, density=0.7, color='k')

if ev.imag[0] == 0 and ev.imag[1] == 0:
    ax.plot(t, (evec[1,0]/evec[0,0]) * t, ls='-')
    ax.plot(t, (evec[1,1]/evec[0,1]) * t, ls='-')

ax.set_xlim([-width, width])
ax.set_ylim([-width, width])
ax.set_xlabel('$g$')
ax.set_ylabel('$y$')
# ここまでは上と同じ。
ax.plot(res.x, calc_ys(y0, res.x))
plt.show()

# + [markdown] id="m-BlSCSI8n2M"
# …ということで、上の緑のギザギザが「力技」なわけだが、あまりうまくいっていない。dt を小さくすればギザギザはとても小さくなりより streamplot の線に沿った形にはなるようだが、いかんせん小さすぎてちゃんとした値になっているかよくわからない。
#
# ただ、この実験で、gs が百のパラメータからなるとても長いベクトルにしたわけだが、それがわりと短い時間で結果が出ている。それは今後にむけて心強い結果だといえる。
