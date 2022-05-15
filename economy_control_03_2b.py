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
# # 村田安雄『動的経済システムの最適制御』第3章2節bの検算
#
# (Version: 0.0.1)
#
# 「第3章 離散時間ダイナミック・プログラミングの方法と消費計画への応用」の「2. 確実性下の D.P. の最適化原理」の「［b］m 制御変数と n 状態変数の D.P.」について、行列に関する計算の「検算」を行ってみる。
#
# これは sympy_matrix_tools のテストでもある。
#
# 本当はこういった証明は専門の theorem prover を使ってやるべきである。しかし、無料で誰でも使える Google Colab & Python で示せればそれはそれで価値があるのではないかと考えた。

# + colab={"base_uri": "https://localhost:8080/"} id="jiU2987p_niD" outputId="6c107652-fe20-495d-a32f-667e733fe8d1"
from time import gmtime, strftime
print("Time-stamp: <%s>" % strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()))

# + colab={"base_uri": "https://localhost:8080/"} id="XEwKNWzrgtk4" outputId="c7b1c38b-7b7b-4032-8973-9432772f192e"
# !pip install git+https://github.com/JRF-2018/sympy_matrix_tools

# + id="BQjUJcHwgxuH"
from sympy import *
init_printing()
from sympy_matrix_tools import *

# + id="Cc90E2exg3yv"
t = Symbol("t", integer=True, positive=True)
n = Symbol("n", integer=True, positive=True)
m = Symbol("m", integer=True, positive=True)
y = MatrixFunction("y", n, 1)
x = MatrixFunction("x", m, 1)
z = MatrixFunction("z", n, 1)
A = MatrixSymbol("A", n, n)
B = MatrixSymbol("B", n, m)

EQ11 = Eq(y(t), A * y(t -1) + B * x(t) + z(t))

# + [markdown] id="S2Iy2mr1srvw"
# なお P, Q はのちに定義する P, Q の MatrixFunction と名前がダブるため P_T, Q_T と名付けておく。 

# + id="BQzBUxAQiys3"
T = Symbol("T", integer=True, positive=True)
tau = Symbol("tau", integer=True, positive=True)
beta = Symbol("beta", real=True)
k = Symbol("k", integer=True, positive=True)

P_T = MatrixSymbol("P_T", n, 1)
Q_T = MatrixSymbol("Q_T", n, n)

EQQT = Eq(Q_T.T, Q_T)

J = MatrixFunction("J", 1, 1)
w = MatrixFunction("w", 1, 1)

EQ12 = Eq(J(T, 1), Sum((beta ** (t - 1)) * w(y(t)), (t, 1, T)))
EQ13 = Eq(w(y(t)), P_T.T * y(t) + Rational(1/2) * y(t).T * Q_T * y(t))
EQ14 = Eq(y(t), A**(t - (tau - 1)) * y(tau - 1)
          + MatSum(A ** (t - k) * (B * x(k) ++ z(k)), (k, tau, t)))

# + [markdown] id="KLy71WbEm3zQ"
# ここで x(tau) の系列に関して式(15)や式(16)を定義するのは面倒なので、直接、式(17)を天下り的に次のように定義したい。しかし、SymPy の Min はその引数に集合をとることができない。

# + id="WVxocKz3kyvx"
# v = MatrixFunction("v", 1, 1)
# x_tau = Symbol("x_tau", real=True)

# EQ17 = Eq(v(tau, y(tau - 1)),
#           Min(ImageSet(Lambda(x_tau, Subs(w(y(tau)) + beta * v(tau + 1, y(tau)),
#                                    x(tau), x_tau)), Interval(0, oo))))

# + [markdown] id="m-b3I3ruqiDG"
# そこで定義できたとしてもやるべきことだった微分のための式のみ定義しておく。

# + id="_NOIv9ufn_XG"
v = MatrixFunction("v", 1, 1)

EQ17_lhs = v(tau, y(tau - 1))
EQ17_rhs = w(y(tau)) + beta * v(tau + 1, y(tau))

# + [markdown] id="UZvF3EHbsCa-"
# さてここから本では明示してないが、帰納法で式を示すことになる。

# + id="RtPtXDnYrB2n"
S = MatrixFunction("S", n, n)
P = MatrixFunction("P", n, 1)
Q = MatrixFunction("Q", n, n)
h = MatrixFunction("h", 1, 1)
g = MatrixFunction("g", 1, 1)

EQ21D = Eq(S(t), Identity(n) - B * ((B.T * Q(t) * B) ** - 1) * B.T * Q(t))
EQ24D = Eq(Q(t), Q_T + beta * A.T * Q(t + 1) * S(t + 1) * A)
EQ23D = Eq(P(t), P_T + beta * A.T * S(t + 1).T * P(t + 1)
           + beta * A.T * Q(t + 1) * S(t + 1) * z(t + 1))
EQ22D = Eq(h(t), P(t).T * B * ((B.T * Q(t) * B) ** -1) * B.T * P(t))
EQ25D = Eq(g(t), beta * (P(t + 1).T * S(t + 1) * z(t + 1)
                         + (Rational(1, 2) * z(t + 1).T * Q(t + 1) * S(t + 1)
                            * z(t + 1))
                         - Rational(1, 2) * h(t + 1) + g(t + 1)))
EQGT = Eq(g(T), ZeroMatrix(1, 1))

EQ19D = Eq(x(t), - ((B.T * Q(t) * B) ** -1) * B.T *
           (Q(t) * (A * y(t - 1) + z(t)) + P(t)))

EQ20D = Eq(v(t, y(t - 1)),
           (P(t).T + Rational(1, 2) * (A * y(t -1) + z(t)).T * Q(t))
           * S(t) * (A * y(t -1) + z(t))
           - Rational(1, 2) * h(t) + g(t))

# + [markdown] id="AiNvQS3Fvh7u"
# 示したいのは t := t において EQ21D～EQ25D のときに EQ19D EQ20D が成り立っていたとき、t := t-1 において、EQ21D～EQ25D と定義したとき EQ19D EQ20D が成り立つことである。また帰納法の初期条件として t := T-1 において EQ21D～EQ25D と定義したとき EQ19D EQ20D が成り立つこともまず示さねばならない。
#
# まずは本にしたがい、t := T-1 の場合を示す。
#
# まず次を x(t) で微分したい。

# + id="3x4SV3h-yNjX"
EQ18a_rhs = P_T.T * y(T) + Rational(1, 2) * y(T).T * Q_T * y(T)

# + colab={"base_uri": "https://localhost:8080/", "height": 58} id="Q52c1UWFy65-" outputId="9c324cef-55bf-4ef3-a45b-0fc84bc29821"
EQ11tmp = EQ11.subs(t, T)
EQ18tmp = EQ18a_rhs.subs(EQ11tmp.lhs, EQ11tmp.rhs)
EQ18tmp

# + [markdown] id="Vjtq-IHfzcQ2"
# MatrixFunction はまだ微分に対応していないのでいったんここに出てくるものを MatrixSymbol に置き換えねばならない。

# + colab={"base_uri": "https://localhost:8080/", "height": 58} id="ih51kChczLY-" outputId="fc656757-ec94-4f21-be2e-8c2d0d962b24"
z_T = MatrixSymbol("z_T", n, 1)
y_Tm1 = MatrixSymbol("y_Tm1", n, 1)
x_T = MatrixSymbol("x_T", m, 1)
EQ18tmp2 = EQ18tmp.subs({x(T): x_T, y(T - 1): y_Tm1, z(T): z_T})
EQ18tmp2

# + colab={"base_uri": "https://localhost:8080/", "height": 38} id="QgTR0jp60sCW" outputId="fcb8e24b-7ec6-4c2d-b259-507e9d192b50"
EQ18tmp3 = EQ18tmp2.diff(x_T)\
                   .subs(EQQT.lhs, EQQT.rhs)\
                   .subs({x_T: x(T), y_Tm1: y(T - 1), z_T: z(T)})\
                   .doit()
EQ18tmp3

# + colab={"base_uri": "https://localhost:8080/", "height": 38} id="VZdvkpdS0wWX" outputId="8fecdd67-4527-41cb-fc83-832f3692ec5e"
EQ18tmp4 = EQ18tmp3.expand().doit()
EQ18tmp4

# + colab={"base_uri": "https://localhost:8080/", "height": 42} id="McOUs36F1hbO" outputId="e5aab21e-7454-43c1-ced2-533fc1f7a88f"
EQ18tmp5 = mat_collect(mat_divide(EQ18tmp4, B.T * Q_T * B), ((B.T * Q_T * B) ** -1) * B.T, right=True)
EQ18tmp5

# + [markdown] id="-zlWWNHC-CMn"
# EQ18tmp5 == 0 と置いて、これが EQ19D と同じことを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="ZAv51P462j9-" outputId="0b12c591-5dea-4f5f-dec1-cc17b96da3d6"
EQ19a = EQ19D.subs(t, T).subs(Q(T), Q_T).subs(P(T), P_T)
(EQ19a.lhs - EQ19a.rhs - EQ18tmp5).expand().doit()

# + [markdown] id="jiCxOcHB9vNt"
# E19a と EQ18tmp5 は同じことであることが示せた。最適値を求めるためにこれを EQ18tmp に代入して整理していく。

# + colab={"base_uri": "https://localhost:8080/", "height": 108} id="uxVZgILd9Rxm" outputId="b337375a-049b-4c6e-d4b2-1919a726a47f"
EQ20tmp = EQ18tmp.subs(EQ19a.lhs, EQ19a.rhs)
EQ20tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 180} id="1BmZmVCu-Arm" outputId="1a18a6b4-a834-46ac-a09b-4766aad8198a"
EQ20tmp2 = mat_trivial_divide(EQ20tmp.expand().subs(EQQT.lhs, EQQT.rhs)).doit()
EQ20tmp2

# + [markdown] id="7p6nKCLz93GX"
# これが EQ20D と等しいことを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 180} id="IB0fIS22VVCS" outputId="19cfe2c2-4503-4ae7-c022-88946f238271"
EQ21a = EQ21D.subs(t, T).subs(Q(T), Q_T)
EQ22a = EQ22D.subs(t, T).subs(Q(T), Q_T).subs(P(T), P_T)
EQ20a = EQ20D.subs(t, T).subs(Q(T), Q_T).subs(P(T), P_T)\
                                        .subs(EQGT.lhs, EQGT.rhs).doit()

EQ20tmp3 = EQ20a.rhs.subs({EQ21a.lhs: EQ21a.rhs, EQ22a.lhs: EQ22a.rhs})\
                    .expand().doit()
EQ20tmp3

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="obekGmrjDSc3" outputId="5070c6db-cf72-4186-a820-2aee7ecd9d59"
(EQ20tmp2 - EQ20tmp3).expand().doit()

# + [markdown] id="bKK740TvK1YM"
# これで式(20a) が正しいことが示せた。今後～tmp 系は再利用していくので混乱しないよう。
#
# 次に式(18b)を示していこう。

# + id="_Ak_u9UkENpN" colab={"base_uri": "https://localhost:8080/", "height": 98} outputId="d6857f87-0ed2-4806-8c49-5b894e77b6d8"
EQ17_lhs_tmp = EQ17_lhs.subs(tau, T - 1)
EQ17_rhs_tmp = EQ17_rhs.subs(tau, T - 1)
EQ13tmp = EQ13.subs(t, T -1)
EQ18tmp = EQ17_rhs_tmp.subs({EQ13tmp.lhs: EQ13tmp.rhs, EQ20a.lhs: EQ20a.rhs})
EQ18tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 58} id="jv39y4EOxe1Y" outputId="a666e322-5091-4bab-bbe3-727c0f4e3405"
EQ13tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 98} id="m15B1exuYz-L" outputId="f766f0fb-4c9d-4f54-9682-ac5f26a45c7c"
EQ18b_rhs = P(T - 1).T * y(T - 1) \
    + Rational(1, 2) * y(T - 1).T * Q(T - 1) * y(T-1) + g(T - 1)
EQ24tmp = EQ24D.subs(t, T - 1)
EQ23tmp = EQ23D.subs(t, T - 1)
EQ25tmp = EQ25D.subs(t, T - 1).subs(EQGT.lhs, EQGT.rhs).doit()
EQ18tmp2 = EQ18b_rhs.subs({EQ24tmp.lhs: EQ24tmp.rhs,
                           EQ23tmp.lhs: EQ23tmp.rhs,
                           EQ25tmp.lhs: EQ25tmp.rhs,
                           Q(T): Q_T,
                           P(T): P_T,
                           EQQT.lhs: EQQT.rhs})
EQ18tmp2

# + colab={"base_uri": "https://localhost:8080/", "height": 98} id="v7aAjf1fah0x" outputId="405465d2-19c5-4114-92c3-3e75c7a94aad"
EQ21tmp = EQ21D.subs(t, T).subs(Q(T), Q_T)
EQ18tmp3 = (EQ18tmp - EQ18tmp2).subs(EQ21tmp.lhs, EQ21tmp.rhs).expand()\
    .subs(EQQT.lhs, EQQT.rhs).doit()
EQ18tmp3

# + [markdown] id="7Dv--QP1mIJp"
# 今次の式が成り立つことから、B の逆行列は取れないが、X == Q_T と仮定する。

# + colab={"base_uri": "https://localhost:8080/", "height": 38} id="nZg9rufLaonL" outputId="c1441260-7506-443c-9707-56bd5107feed"
X = MatrixSymbol("X", n, n)
EQtmp = Eq(X, (Q_T * B * (B.T * Q_T * B) ** -1 * B.T * Q_T))
EQtmp2 = Eq(EQtmp.lhs * B, mat_trivial_divide(EQtmp.rhs * B))
EQtmp2

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="woix3CP0mV1w" outputId="068c49bd-bb63-4199-d80d-bdc29a517950"
EQ18tmp3.subs(EQtmp.rhs, Q_T).doit()

# + [markdown] id="-PGUpDkKmmYC"
# ちょっと裏技っぽいのを使ったが、式(18b)は示せたものとする。

# + colab={"base_uri": "https://localhost:8080/", "height": 80} id="JRzTUN_wj51x" outputId="2a4d0f82-fdfa-4a9a-aa10-745c90532d2a"
EQ11tmp = EQ11.subs(t, T - 1)
EQ18tmp = EQ18b_rhs.subs(EQ11tmp.lhs, EQ11tmp.rhs)
EQ18tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 58} id="JVcqx5h08Xeu" outputId="efac4907-8adc-40b5-8b1f-59225f863712"
z_Tm1 = MatrixSymbol("z_Tm1", n, 1)
y_Tm2 = MatrixSymbol("y_Tm2", n, 1)
x_Tm1 = MatrixSymbol("x_Tm1", m, 1)
P_Tm1 = MatrixSymbol("P_Tm1", n, 1)
Q_Tm1 = MatrixSymbol("Q_Tm1", n, n)
g_Tm1 = MatrixSymbol("g_Tm1", 1, 1)
EQ18tmp2 = EQ18tmp.subs({x(T - 1): x_Tm1,
                         y(T - 2): y_Tm2,
                         z(T - 1): z_Tm1,
                         P(T - 1): P_Tm1,
                         Q(T - 1): Q_Tm1,
                         g(T - 1): g_Tm1})
EQ18tmp3 = EQ18tmp2.diff(x_Tm1)
EQ18tmp4 = EQ18tmp3.subs({x_Tm1: x(T - 1),
                          y_Tm2: y(T - 2),
                          z_Tm1: z(T - 1),
                          P_Tm1: P(T - 1),
                          Q_Tm1: Q(T - 1),
                          g_Tm1: g(T - 1)})
EQ18tmp4

# + [markdown] id="LMvifeRwqbso"
# さてここで帰納法で Q(t).T == Q(t) を示す。t := T のときは言えているので、Q(t + 1).T == Q(t + 1) が言えているとして Q(t).T == Q(t) を示す。

# + id="YgRHd_tvnrEI"
EQ21tmp = EQ21D.subs(t, t + 1)
EQ24tmp = EQ24D.subs(EQ21tmp.lhs, EQ21tmp.rhs).expand().doit()

# + colab={"base_uri": "https://localhost:8080/", "height": 42} id="8jeNN4t4qJJB" outputId="97cab424-5f72-4cbf-81d1-10d9379abf73"
EQ24tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="GpJOatscqLDC" outputId="69b80b2c-915b-4fa7-fb2c-db8b2b4c68a7"
EQ24tmp2 = EQ24tmp.rhs.T.subs(Q(t + 1).T, Q(t + 1)).subs(EQQT.lhs, EQQT.rhs).doit()
(EQ24tmp.rhs - EQ24tmp2).expand().doit()

# + [markdown] id="7jlKw1E6rjcg"
# ゆえに次が言えた。

# + id="G5geS7r_qOaB"
EQQtT = Eq(Q(t).T, Q(t))

# + [markdown] id="_Qoahu_vrps3"
# 微分に戻ろう。

# + colab={"base_uri": "https://localhost:8080/", "height": 38} id="E-PyiEoAsP0Y" outputId="b7906827-0016-4902-d35a-978692e55ba8"
EQQtTtmp = EQQtT.subs(t, T - 1)
EQ18tmp5 = EQ18tmp4.subs(EQQtTtmp.lhs, EQQtTtmp.rhs).expand().doit()
EQ18tmp5

# + colab={"base_uri": "https://localhost:8080/", "height": 42} id="dCw5vn_lsfSR" outputId="d5c3a1e6-a0d8-4205-8c98-6068b9d59b85"
EQ18tmp6 = mat_collect(mat_divide(EQ18tmp5, B.T * Q(T - 1) * B), ((B.T * Q(T - 1) * B) ** -1) * B.T, right=True)
EQ18tmp6

# + [markdown] id="vOq4YPfa-Q6-"
# EQ18tmp6 == 0 と置いて、これが EQ19D と同じことを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="gJL7scS_tA44" outputId="ff1674e4-f302-494c-c575-6f434ebfabbb"
EQ19b = EQ19D.subs(t, T - 1)
(EQ19b.lhs - EQ19b.rhs - EQ18tmp6).expand().doit()

# + [markdown] id="KxO66sfmyloX"
# E19b と EQ18tmp6 は同じことであることが示せた。これを最適値を求めるため EQ18tmp に代入して整理していく。

# + colab={"base_uri": "https://localhost:8080/", "height": 129} id="mVeZOTNntRP4" outputId="6f8220a4-58c0-4a70-b6e2-0d577c47f23f"
EQ20tmp = EQ18tmp.subs(EQ19b.lhs, EQ19b.rhs)
EQ20tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 286} id="DcPYXagpzQmv" outputId="72cf35f2-1ed6-4c0c-a352-e13ed0a5d292"
EQQtTtmp = EQQtT.subs(t, T - 1)
EQ20tmp2 = mat_trivial_divide(EQ20tmp.expand().subs(EQQtTtmp.lhs, EQQtTtmp.rhs)).doit()
EQ20tmp2

# + [markdown] id="Z0RobKK69tnn"
# これが EQ20D と同じであることを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 286} id="US7eVPyLzkaw" outputId="6da8ba27-9248-4892-aa11-ae2612309692"
EQ21tmp = EQ21D.subs(t, T - 1)
EQ22tmp = EQ22D.subs(t, T - 1)
EQ20tmp = EQ20D.subs(t, T - 1)

EQ20tmp3 = EQ20tmp.rhs.subs({EQ21tmp.lhs: EQ21tmp.rhs,
                             EQ22tmp.lhs: EQ22tmp.rhs})\
                      .expand().doit()
EQ20tmp3

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="UCfwnvsV0fZQ" outputId="d68cfd8b-f329-4fd1-d611-9f977f904cba"
(EQ20tmp2 - EQ20tmp3).expand().doit()

# + [markdown] id="lQ-T_X2n0uO3"
# これで t := T - 1 において、EQ19D EQ20D が成り立つことが示せたことになる。
#
# 次に帰納法の本題である。t := t において EQ19D EQ20D が成り立っているとき、t := t - 1 において EQ19D EQ20D が成り立っていることを示す。
#

# + colab={"base_uri": "https://localhost:8080/", "height": 98} id="xgxfu3Xd0ofA" outputId="109f7514-47fe-4f84-a928-b9045f81aaa7"
EQ17_lhs_tmp = EQ17_lhs.subs(tau, t - 1)
EQ17_rhs_tmp = EQ17_rhs.subs(tau, t - 1)
EQ13tmp = EQ13.subs(t, t -1)
EQ18tmp = EQ17_rhs_tmp.subs({EQ13tmp.lhs: EQ13tmp.rhs, EQ20D.lhs: EQ20D.rhs})
EQ18tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 98} id="x_LH6F8x18Lo" outputId="64583129-646d-4dbb-cf85-dacf35c64ae4"
EQ18c_rhs = P(t - 1).T * y(t - 1) \
    + Rational(1, 2) * y(t - 1).T * Q(t - 1) * y(t-1) + g(t - 1)
EQ24tmp = EQ24D.subs(t, t - 1)
EQ23tmp = EQ23D.subs(t, t - 1)
EQ25tmp = EQ25D.subs(t, t - 1).subs(EQGT.lhs, EQGT.rhs).doit()
EQ18tmp2 = EQ18c_rhs.subs({EQ24tmp.lhs: EQ24tmp.rhs,
                           EQ23tmp.lhs: EQ23tmp.rhs,
                           EQ25tmp.lhs: EQ25tmp.rhs})
EQ18tmp2

# + colab={"base_uri": "https://localhost:8080/", "height": 98} id="wMVZFKi_2SPo" outputId="608e9292-1160-40c6-a97c-c2a09088d2f3"
EQ18tmp3 = (EQ18tmp - EQ18tmp2)\
    .subs(EQ21D.lhs, EQ21D.rhs).expand()\
    .subs(EQQtT.lhs, EQQtT.rhs).expand().doit()
EQ18tmp3

# + [markdown] id="dF1IHJG53tXK"
# ここで上の X のハックと同じものが成立しているとする。

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="3T0s6Z_H3Jqg" outputId="8abf7947-cfb5-4ec3-b82a-847a7a4008ca"
EQ18tmp3.subs((Q(t) * B * (B.T * Q(t) * B) ** -1 * B.T * Q(t)), Q(t)).doit()

# + colab={"base_uri": "https://localhost:8080/", "height": 80} id="nfCo2wQ74Bqh" outputId="bc78fc2a-0eab-497a-db79-c8f769272d53"
EQ11tmp = EQ11.subs(t, t - 1)
EQ18tmp = EQ18c_rhs.subs(EQ11tmp.lhs, EQ11tmp.rhs)
EQ18tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 58} id="CBiCbMWf8Ljf" outputId="d64b71e7-7f58-447b-f770-adecd61babeb"
z_tm1 = MatrixSymbol("z_tm1", n, 1)
y_tm2 = MatrixSymbol("y_tm2", n, 1)
x_tm1 = MatrixSymbol("x_tm1", m, 1)
P_tm1 = MatrixSymbol("P_tm1", n, 1)
Q_tm1 = MatrixSymbol("Q_tm1", n, n)
g_tm1 = MatrixSymbol("g_tm1", 1, 1)
EQ18tmp2 = EQ18tmp.subs({x(t - 1): x_tm1,
                         y(t - 2): y_tm2,
                         z(t - 1): z_tm1,
                         P(t - 1): P_tm1,
                         Q(t - 1): Q_tm1,
                         g(t - 1): g_tm1})
EQ18tmp3 = EQ18tmp2.diff(x_tm1)
EQ18tmp4 = EQ18tmp3.subs({x_tm1: x(t - 1),
                          y_tm2: y(t - 2),
                          z_tm1: z(t - 1),
                          P_tm1: P(t - 1),
                          Q_tm1: Q(t - 1),
                          g_tm1: g(t - 1)})
EQ18tmp4

# + colab={"base_uri": "https://localhost:8080/", "height": 38} id="BmKhRFBn4hv_" outputId="5259770b-9055-423f-b1a1-1fe63e7e9ae1"
EQQtTtmp = EQQtT.subs(t, t - 1)
EQ18tmp5 = EQ18tmp4.subs(EQQtTtmp.lhs, EQQtTtmp.rhs).expand().doit()
EQ18tmp5

# + colab={"base_uri": "https://localhost:8080/", "height": 42} id="UeCIweRR5DW4" outputId="fdb62bf2-e24b-4080-c235-1bbd83569bcc"
EQ18tmp6 = mat_collect(mat_divide(EQ18tmp5, B.T * Q(t - 1) * B), ((B.T * Q(t - 1) * B) ** -1) * B.T, right=True)
EQ18tmp6

# + [markdown] id="BWWRwatk-hWH"
# EQ18tmp6 == 0 と置いて、これが EQ19D と同じことを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="jkIhAfFA5Hq3" outputId="ec540b35-39fe-41da-e6f0-315cfbdb50a2"
EQ19c = EQ19D.subs(t, t - 1)
(EQ19c.lhs - EQ19c.rhs - EQ18tmp6).expand().doit()

# + [markdown] id="Uo-gL8685ZDO"
# E19c と EQ18tmp6 は同じことであることが示せた。これを最適値を求めるため EQ18tmp に代入して整理していく。

# + colab={"base_uri": "https://localhost:8080/", "height": 129} id="HQzrLH9J5doH" outputId="3af66d79-db4b-4ff7-c7be-ec0a094e14cb"
EQ20tmp = EQ18tmp.subs(EQ19c.lhs, EQ19c.rhs)
EQ20tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 286} id="PbNsdaqE5gxA" outputId="43f89f4a-71fb-4635-91dc-a5994529b19a"
EQQtTtmp = EQQtT.subs(t, t - 1)
EQ20tmp2 = mat_trivial_divide(EQ20tmp.expand().subs(EQQtTtmp.lhs, EQQtTtmp.rhs)).doit()
EQ20tmp2

# + [markdown] id="mpNjuwwo-tFm"
# これが EQ20D と同じことを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 286} id="5yezUb-D5ltZ" outputId="cf180f2c-e91d-4f1a-9dc6-35c9b6704b7d"
EQ21tmp = EQ21D.subs(t, t - 1)
EQ22tmp = EQ22D.subs(t, t - 1)
EQ20tmp = EQ20D.subs(t, t - 1)

EQ20tmp3 = EQ20tmp.rhs.subs({EQ21tmp.lhs: EQ21tmp.rhs,
                             EQ22tmp.lhs: EQ22tmp.rhs})\
                      .expand().doit()
EQ20tmp3

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="4V7g2YO15vpo" outputId="8d464640-bc84-4c13-99f8-78876c324999"
(EQ20tmp2 - EQ20tmp3).expand().doit()

# + [markdown] id="5OJkwQZ459Xi"
# これで t := t - 1 において、EQ19D EQ20D が成り立つことが示せたことになる。これで帰納法でこの節の導出が正しかったことが証明された。…と思う。
#
