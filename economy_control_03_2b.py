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
# (Version: 0.0.2)
#
# 「第3章 離散時間ダイナミック・プログラミングの方法と消費計画への応用」の「2. 確実性下の D.P. の最適化原理」の「［b］m 制御変数と n 状態変数の D.P.」について、行列に関する計算の「検算」を行ってみる。
#
# これは sympy_matrix_tools のテストでもある。
#
# 本当はこういった証明は専門の theorem prover を使ってやるべきである。しかし、無料で誰でも使える Google Colab & Python で示せればそれはそれで価値があるのではないかと考えた。

# + colab={"base_uri": "https://localhost:8080/"} id="jiU2987p_niD" outputId="071a1de4-ffd8-4a01-b9a5-8b5de22b049c"
from time import gmtime, strftime
print("Time-stamp: <%s>" % strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()))

# + colab={"base_uri": "https://localhost:8080/"} id="XEwKNWzrgtk4" outputId="5c5cd8e7-a9ed-4e62-aacf-c8956af2f9dc"
# !pip install git+https://github.com/JRF-2018/sympy_matrix_tools

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="bjEwAq1cbaJw" outputId="79286c00-9966-472e-dba7-05d39365d484"
import sympy
sympy.__version__

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="LEmmj-7abNAY" outputId="791a58ac-04b6-419b-de9b-cebaa98a5b1f"
import sympy_matrix_tools
sympy_matrix_tools.__version__

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

EQQTT = Eq(Q_T.T, Q_T)

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

# + colab={"base_uri": "https://localhost:8080/", "height": 58} id="Q52c1UWFy65-" outputId="7e63d075-9726-4826-d27b-4ab06df9acda"
EQ11tmp = EQ11.subs(t, T)
EQ18tmp = EQ18a_rhs.subs(EQ11tmp.lhs, EQ11tmp.rhs)
EQ18tmp

# + [markdown] id="Vjtq-IHfzcQ2"
# MatrixFunction はまだ微分に対応していないのでいったんここに出てくるものを MatrixSymbol に置き換えねばならない。

# + colab={"base_uri": "https://localhost:8080/", "height": 58} id="ih51kChczLY-" outputId="dd9f48e5-bba8-432f-dfdd-3ef46d769ccb"
z_T = MatrixSymbol("z_T", n, 1)
y_Tm1 = MatrixSymbol("y_Tm1", n, 1)
x_T = MatrixSymbol("x_T", m, 1)
EQ18tmp2 = EQ18tmp.subs({x(T): x_T, y(T - 1): y_Tm1, z(T): z_T})
EQ18tmp2

# + colab={"base_uri": "https://localhost:8080/", "height": 38} id="QgTR0jp60sCW" outputId="90d6d582-f03a-4202-fd3d-27b6aecd5368"
EQ18tmp3 = EQ18tmp2.diff(x_T)\
                   .subs(EQQTT.lhs, EQQTT.rhs)\
                   .subs({x_T: x(T), y_Tm1: y(T - 1), z_T: z(T)})\
                   .doit()
EQ18tmp3

# + colab={"base_uri": "https://localhost:8080/", "height": 38} id="VZdvkpdS0wWX" outputId="936ceb02-1eb5-4444-a503-64260016dea9"
EQ18tmp4 = EQ18tmp3.expand().doit()
EQ18tmp4

# + colab={"base_uri": "https://localhost:8080/", "height": 42} id="McOUs36F1hbO" outputId="56c74360-89be-4806-dc9d-c79856b9ea66"
EQ18tmp5 = mat_collect(mat_divide(EQ18tmp4, B.T * Q_T * B), ((B.T * Q_T * B) ** -1) * B.T, right=True)
EQ18tmp5

# + [markdown] id="-zlWWNHC-CMn"
# EQ18tmp5 == 0 と置いて、これが EQ19D と同じことを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="ZAv51P462j9-" outputId="b099f16b-57d9-40d6-8f24-a7a8272ae285"
EQ19a = EQ19D.subs(t, T).subs(Q(T), Q_T).subs(P(T), P_T)
(EQ19a.lhs - EQ19a.rhs - EQ18tmp5).expand().doit()

# + [markdown] id="jiCxOcHB9vNt"
# E19a と EQ18tmp5 は同じことであることが示せた。最適値を求めるためにこれを EQ18tmp に代入して整理していく。

# + colab={"base_uri": "https://localhost:8080/", "height": 108} id="uxVZgILd9Rxm" outputId="fdd56646-e646-435d-9742-ad2a880e56a7"
EQ20tmp = EQ18tmp.subs(EQ19a.lhs, EQ19a.rhs)
EQ20tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 180} id="1BmZmVCu-Arm" outputId="73e07e3b-cf6f-40a9-a8d0-d8e0fbe97044"
EQ20tmp2 = mat_trivial_divide(EQ20tmp.expand().subs(EQQTT.lhs, EQQTT.rhs)).doit()
EQ20tmp2

# + [markdown] id="7p6nKCLz93GX"
# これが EQ20D と等しいことを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 180} id="IB0fIS22VVCS" outputId="13d80802-0c7e-4cdd-b079-f806eeadd7f6"
EQ21a = EQ21D.subs(t, T).subs(Q(T), Q_T)
EQ22a = EQ22D.subs(t, T).subs(Q(T), Q_T).subs(P(T), P_T)
EQ20a = EQ20D.subs(t, T).subs(Q(T), Q_T).subs(P(T), P_T)\
                                        .subs(EQGT.lhs, EQGT.rhs).doit()

EQ20tmp3 = EQ20a.rhs.subs({EQ21a.lhs: EQ21a.rhs, EQ22a.lhs: EQ22a.rhs})\
                    .expand().doit()
EQ20tmp3

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="obekGmrjDSc3" outputId="4914e755-d605-45b5-9bb4-78fa63ff9f72"
(EQ20tmp2 - EQ20tmp3).expand().doit()

# + [markdown] id="bKK740TvK1YM"
# これで式(20a) が正しいことが示せた。今後～tmp 系は再利用していくので混乱しないよう。
#
# 次に式(18b)を示していこう。

# + id="_Ak_u9UkENpN" colab={"base_uri": "https://localhost:8080/", "height": 98} outputId="fd2f7cd6-2cc3-438f-92b7-35b28c1b761c"
EQ17_lhs_tmp = EQ17_lhs.subs(tau, T - 1)
EQ17_rhs_tmp = EQ17_rhs.subs(tau, T - 1)
EQ13tmp = EQ13.subs(t, T -1)
EQ18tmp = EQ17_rhs_tmp.subs({EQ13tmp.lhs: EQ13tmp.rhs, EQ20a.lhs: EQ20a.rhs})
EQ18tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 72} id="jv39y4EOxe1Y" outputId="5c4851c1-ace3-469f-d6e6-eaf198f6176f"
EQ13tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 98} id="m15B1exuYz-L" outputId="35599c59-e48b-4418-ec3f-82d5ef8a5fbc"
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
                           EQQTT.lhs: EQQTT.rhs})
EQ18tmp2

# + colab={"base_uri": "https://localhost:8080/", "height": 98} id="v7aAjf1fah0x" outputId="b99c679d-14e5-4042-b971-42b61a5f6c22"
EQ21tmp = EQ21D.subs(t, T).subs(Q(T), Q_T)
EQ18tmp3 = (EQ18tmp - EQ18tmp2).subs(EQ21tmp.lhs, EQ21tmp.rhs).expand()\
    .subs(EQQTT.lhs, EQQTT.rhs).doit()
EQ18tmp3

# + [markdown] id="7Dv--QP1mIJp"
# 今次の式が成り立つことから、B の逆行列は取れないが、X == Q_T と仮定する。

# + colab={"base_uri": "https://localhost:8080/", "height": 38} id="nZg9rufLaonL" outputId="1a9a265f-fa98-4691-eed9-f1cc55d6943a"
X = MatrixSymbol("X", n, n)
EQtmp = Eq(X, (Q_T * B * (B.T * Q_T * B) ** -1 * B.T * Q_T))
EQtmp2 = Eq(EQtmp.lhs * B, mat_trivial_divide(EQtmp.rhs * B))
EQtmp2

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="woix3CP0mV1w" outputId="4275039d-e28d-47a4-d67c-743b4d8b3dd5"
EQ18tmp3.subs(EQtmp.rhs, Q_T).doit()

# + [markdown] id="-PGUpDkKmmYC"
# ちょっと裏技っぽいのを使ったが、式(18b)は示せたものとする。

# + colab={"base_uri": "https://localhost:8080/", "height": 80} id="JRzTUN_wj51x" outputId="3506aa20-5c08-4620-8ca8-0ee7a67cd15e"
EQ11tmp = EQ11.subs(t, T - 1)
EQ18tmp = EQ18b_rhs.subs(EQ11tmp.lhs, EQ11tmp.rhs)
EQ18tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 58} id="JVcqx5h08Xeu" outputId="3bd98e8e-bca2-4535-f5b0-8cf75cb8a40e"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 42} id="8jeNN4t4qJJB" outputId="dd47155e-ac7d-45be-dd59-ed764d3c940a"
EQ24tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="GpJOatscqLDC" outputId="ff7a8bd7-8062-41fa-e779-1148e4dca8cf"
EQ24tmp2 = EQ24tmp.rhs.T.subs(Q(t + 1).T, Q(t + 1)).subs(EQQTT.lhs, EQQTT.rhs).doit()
(EQ24tmp.rhs - EQ24tmp2).expand().doit()

# + [markdown] id="7jlKw1E6rjcg"
# ゆえに次が言えた。

# + id="G5geS7r_qOaB"
EQQtT = Eq(Q(t).T, Q(t))

# + [markdown] id="_Qoahu_vrps3"
# 微分に戻ろう。

# + colab={"base_uri": "https://localhost:8080/", "height": 38} id="E-PyiEoAsP0Y" outputId="7a6b8707-92cb-4216-982c-00cb1b82a829"
EQQtTtmp = EQQtT.subs(t, T - 1)
EQ18tmp5 = EQ18tmp4.subs(EQQtTtmp.lhs, EQQtTtmp.rhs).expand().doit()
EQ18tmp5

# + colab={"base_uri": "https://localhost:8080/", "height": 42} id="dCw5vn_lsfSR" outputId="8fec66e3-7f6f-441b-c0f9-571862b8ee57"
EQ18tmp6 = mat_collect(mat_divide(EQ18tmp5, B.T * Q(T - 1) * B), ((B.T * Q(T - 1) * B) ** -1) * B.T, right=True)
EQ18tmp6

# + [markdown] id="vOq4YPfa-Q6-"
# EQ18tmp6 == 0 と置いて、これが EQ19D と同じことを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="gJL7scS_tA44" outputId="72de4b6e-251b-4170-b596-ce0fdfdefba2"
EQ19b = EQ19D.subs(t, T - 1)
(EQ19b.lhs - EQ19b.rhs - EQ18tmp6).expand().doit()

# + [markdown] id="KxO66sfmyloX"
# E19b と EQ18tmp6 は同じことであることが示せた。これを最適値を求めるため EQ18tmp に代入して整理していく。

# + colab={"base_uri": "https://localhost:8080/", "height": 129} id="mVeZOTNntRP4" outputId="48a2e18c-d7b3-4825-dd96-7240e01453e4"
EQ20tmp = EQ18tmp.subs(EQ19b.lhs, EQ19b.rhs)
EQ20tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 286} id="DcPYXagpzQmv" outputId="af6ccba2-e365-4473-a787-0ba225b4e6d9"
EQQtTtmp = EQQtT.subs(t, T - 1)
EQ20tmp2 = mat_trivial_divide(EQ20tmp.expand().subs(EQQtTtmp.lhs, EQQtTtmp.rhs)).doit()
EQ20tmp2

# + [markdown] id="Z0RobKK69tnn"
# これが EQ20D と同じであることを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 286} id="US7eVPyLzkaw" outputId="3008605c-94d9-48bf-e367-798787db3308"
EQ21tmp = EQ21D.subs(t, T - 1)
EQ22tmp = EQ22D.subs(t, T - 1)
EQ20tmp = EQ20D.subs(t, T - 1)

EQ20tmp3 = EQ20tmp.rhs.subs({EQ21tmp.lhs: EQ21tmp.rhs,
                             EQ22tmp.lhs: EQ22tmp.rhs})\
                      .expand().doit()
EQ20tmp3

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="UCfwnvsV0fZQ" outputId="c453d69e-bf00-4907-8c1d-892874ca7c14"
(EQ20tmp2 - EQ20tmp3).expand().doit()

# + [markdown] id="lQ-T_X2n0uO3"
# これで t := T - 1 において、EQ19D EQ20D が成り立つことが示せたことになる。
#
# 次に帰納法の本題である。t := t において EQ19D EQ20D が成り立っているとき、t := t - 1 において EQ19D EQ20D が成り立っていることを示す。
#

# + colab={"base_uri": "https://localhost:8080/", "height": 98} id="xgxfu3Xd0ofA" outputId="e5057d5f-e410-4fed-a178-f0e279dc9f53"
EQ17_lhs_tmp = EQ17_lhs.subs(tau, t - 1)
EQ17_rhs_tmp = EQ17_rhs.subs(tau, t - 1)
EQ13tmp = EQ13.subs(t, t -1)
EQ18tmp = EQ17_rhs_tmp.subs({EQ13tmp.lhs: EQ13tmp.rhs, EQ20D.lhs: EQ20D.rhs})
EQ18tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 98} id="x_LH6F8x18Lo" outputId="6b76890e-d075-449e-92f7-43a752105a32"
EQ18c_rhs = P(t - 1).T * y(t - 1) \
    + Rational(1, 2) * y(t - 1).T * Q(t - 1) * y(t-1) + g(t - 1)
EQ24tmp = EQ24D.subs(t, t - 1)
EQ23tmp = EQ23D.subs(t, t - 1)
EQ25tmp = EQ25D.subs(t, t - 1).subs(EQGT.lhs, EQGT.rhs).doit()
EQ18tmp2 = EQ18c_rhs.subs({EQ24tmp.lhs: EQ24tmp.rhs,
                           EQ23tmp.lhs: EQ23tmp.rhs,
                           EQ25tmp.lhs: EQ25tmp.rhs})
EQ18tmp2

# + colab={"base_uri": "https://localhost:8080/", "height": 98} id="wMVZFKi_2SPo" outputId="bf197a51-ba9e-45f4-b0d8-6f720c0c62cc"
EQ18tmp3 = (EQ18tmp - EQ18tmp2)\
    .subs(EQ21D.lhs, EQ21D.rhs).expand()\
    .subs(EQQtT.lhs, EQQtT.rhs).expand().doit()
EQ18tmp3

# + [markdown] id="dF1IHJG53tXK"
# ここで上の X のハックと同じものが成立しているとする。

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="3T0s6Z_H3Jqg" outputId="26439c5c-361f-4728-d957-593e2421c953"
EQ18tmp3.subs((Q(t) * B * (B.T * Q(t) * B) ** -1 * B.T * Q(t)), Q(t)).doit()

# + colab={"base_uri": "https://localhost:8080/", "height": 80} id="nfCo2wQ74Bqh" outputId="eaf8c0c5-9eaf-45b7-babd-0ec0b5ffacd1"
EQ11tmp = EQ11.subs(t, t - 1)
EQ18tmp = EQ18c_rhs.subs(EQ11tmp.lhs, EQ11tmp.rhs)
EQ18tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 58} id="CBiCbMWf8Ljf" outputId="4e9dbb73-0ab6-46bb-8501-98ac92f35342"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 38} id="BmKhRFBn4hv_" outputId="d4bf71ac-0bc4-42c0-8780-a0733ce9f5ae"
EQQtTtmp = EQQtT.subs(t, t - 1)
EQ18tmp5 = EQ18tmp4.subs(EQQtTtmp.lhs, EQQtTtmp.rhs).expand().doit()
EQ18tmp5

# + colab={"base_uri": "https://localhost:8080/", "height": 42} id="UeCIweRR5DW4" outputId="75b2673d-d01b-4125-fcf9-134250533ef3"
EQ18tmp6 = mat_collect(mat_divide(EQ18tmp5, B.T * Q(t - 1) * B), ((B.T * Q(t - 1) * B) ** -1) * B.T, right=True)
EQ18tmp6

# + [markdown] id="BWWRwatk-hWH"
# EQ18tmp6 == 0 と置いて、これが EQ19D と同じことを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="jkIhAfFA5Hq3" outputId="3d6a84e4-de9a-4fdc-862a-0469f7d6fdc3"
EQ19c = EQ19D.subs(t, t - 1)
(EQ19c.lhs - EQ19c.rhs - EQ18tmp6).expand().doit()

# + [markdown] id="Uo-gL8685ZDO"
# E19c と EQ18tmp6 は同じことであることが示せた。これを最適値を求めるため EQ18tmp に代入して整理していく。

# + colab={"base_uri": "https://localhost:8080/", "height": 129} id="HQzrLH9J5doH" outputId="110ae91c-b6aa-47fb-ac37-8d75b9bea76d"
EQ20tmp = EQ18tmp.subs(EQ19c.lhs, EQ19c.rhs)
EQ20tmp

# + colab={"base_uri": "https://localhost:8080/", "height": 286} id="PbNsdaqE5gxA" outputId="46df7844-8633-4140-f02f-41cecdd9e021"
EQQtTtmp = EQQtT.subs(t, t - 1)
EQ20tmp2 = mat_trivial_divide(EQ20tmp.expand().subs(EQQtTtmp.lhs, EQQtTtmp.rhs)).doit()
EQ20tmp2

# + [markdown] id="mpNjuwwo-tFm"
# これが EQ20D と同じことを示したい。

# + colab={"base_uri": "https://localhost:8080/", "height": 286} id="5yezUb-D5ltZ" outputId="4c10d0f4-8132-49f5-9e07-b66ab17da41d"
EQ21tmp = EQ21D.subs(t, t - 1)
EQ22tmp = EQ22D.subs(t, t - 1)
EQ20tmp = EQ20D.subs(t, t - 1)

EQ20tmp3 = EQ20tmp.rhs.subs({EQ21tmp.lhs: EQ21tmp.rhs,
                             EQ22tmp.lhs: EQ22tmp.rhs})\
                      .expand().doit()
EQ20tmp3

# + colab={"base_uri": "https://localhost:8080/", "height": 37} id="4V7g2YO15vpo" outputId="933fc144-2cfc-401e-c468-a63415750de3"
(EQ20tmp2 - EQ20tmp3).expand().doit()

# + [markdown] id="5OJkwQZ459Xi"
# これで t := t - 1 において、EQ19D EQ20D が成り立つことが示せたことになる。これで帰納法でこの節の導出が正しかったことが証明された。…と思う。
#
# なお、終わってから気づいたのだが、t := T の時点で帰納法を満たしており、t := T - 1 については言う必要がなかったのかもしれない。ただ、本では示しているため、言った意味が全くないとはならないとは思う。
