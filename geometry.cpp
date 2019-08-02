typedef db T;
//const db eps = 1e-7, pi = acosl(-1.);
int sgn(T x) { return (x > eps) - (x < -eps); }
bool inMid(T l, T x, T r) { return sgn(l - x) * sgn(r - x) <= 0; }  // 数 x 在区间 [l, r] 内（包括边界）

// 点、向量
struct P {
    T x, y; P () {} P(T x, T y) : x(x), y(y) {}
    P operator + (const P &b) const { return P(x + b.x, y + b.y); }  // 向量加
    P operator - (const P &b) const { return P(x - b.x, y - b.y); }  // 向量减
    T operator * (const P &b) const { return x * b.x + y * b.y; }    // 向量点积
    T operator / (const P &b) const { return x * b.y - y * b.x; }    // 向量叉积
    P operator * (const T &k) const { return P(x * k, y * k); }      // 向量数乘
    P operator / (const T &k) const { return P(x / k, y / k); }      // 向量数除
    bool operator < (const P &b) const { return sgn(x - b.x) ? x < b.x : y < b.y; }  // 左小右大，下小上大
    bool operator == (const P &b) const { return !sgn(x - b.x) && !sgn(y - b.y); }   // 向量相等
    P rot(T t) { return P(x * cos(t) - y * sin(t), x * sin(t) + y * cos(t)); }  // 向量逆时针旋转 t 弧度
    P rot90() { return P(-y, x); }           // 向量逆时针旋转 90 度
    db arg() const { return atan2(y, x); }   // 方位角 (-pi, pi]
    T abs() { return sqrt(x * x + y * y); }  // 向量模长
    T abs2() { return x * x + y * y; }       // 向量模长的平方
    P unit() { return (*this) / abs(); }     // 单位向量
    void scan() { db tx, ty; scanf("%lf%lf", &tx, &ty); x = tx; y = ty; }  // 输入
    void print() { printf("%.11lf %.11lf\n", x, y); }                      // 输出
    P toR() { return (sgn(x) == -1 || !sgn(x) && sgn(y) == -1) ? (*this)*(-1) : (*this); }  // 若向量在 2 3 象限，则取相反向量
    bool isUp() const { return sgn(y) == 1 || !sgn(y) && sgn(x) == -1; }                    // 向量是否在 1 2 象限
};
T dis(P a, P b) { return (b - a).abs(); }   // 点 a 到点 b 的距离
P symmetry(P p, P a) { return a + a - p; }  // 点 p 关于点 a 的中心对称点
T cross(P o, P a, P b) { return (a - o) / (b - o); }         // 向量 oa 与 ob 的叉积
int crossSgn(P o, P a, P b) { return sgn(cross(o, a, b)); }  // 向量 oa 与 ob 的叉积符号
db rad(P a, P b) { return atan2(a / b, a * b); }             // 向量 a 与 b 的夹角（弧度制）
bool order(const P &a, const P &b) { return a.isUp() < b.isUp() || a.isUp() == b.isUp() && sgn(a/b) > 0; }  // 极角排序

// 线段、直线
struct L { P s, t; L () {} L(P s, P t) : s(s), t(t) {} };
bool onPS(P p, L a) { return sgn((a.t-a.s)/(p-a.s))==0 && sgn((p-a.s)*(p-a.t))<=0; }   // 点 p 是否在线段 a 上
bool inRec(P p, L a) { return inMid(a.s.x, p.x, a.t.x) && inMid(a.s.y, p.y, a.t.y); }  // 点 p 在以直线 a 为对角线的矩形内
P proj(P p, L a) { return (a.t-a.s) * ( (p-a.s) * (a.t-a.s) / (a.t-a.s).abs2() ) + a.s; }  // 点 p 关于直线 a 的投影点
P reflect(P p, L a) { return proj(p, a) * 2 - p; }                                         // 点 p 关于直线 a 的对称点
bool xSSr(const L &a, const L &b) {  // 线段 a 和线段 b 严格相交
    T c1 = (a.t-a.s) / (b.s-a.s), c2 = (a.t-a.s) / (b.t-a.s);
    T c3 = (b.t-b.s) / (a.s-b.s), c4 = (b.t-b.s) / (a.t-b.s);
    return sgn(c1) * sgn(c2) < 0 && sgn(c3) * sgn(c4) < 0;
}
bool xSS(L a, L b) {                 // 线段 a 和线段 b 不严格相交
    T c1 = (a.t-a.s) / (b.s-a.s), c2 = (a.t-a.s) / (b.t-a.s);
    T c3 = (b.t-b.s) / (a.s-b.s), c4 = (b.t-b.s) / (a.t-b.s);
    return sgn(c1) * sgn(c2) <= 0 && sgn(c3) * sgn(c4) <= 0 &&
           sgn(max(a.s.x,a.t.x) - min(b.s.x,b.t.x)) >= 0 &&
           sgn(max(b.s.x,b.t.x) - min(a.s.x,a.t.x)) >= 0 &&
           sgn(max(a.s.y,a.t.y) - min(b.s.y,b.t.y)) >= 0 &&
           sgn(max(b.s.y,b.t.y) - min(a.s.y,a.t.y)) >= 0;
}
bool xSL(L a, L b) {  // 线段 a 与直线 b 相交 ( 不严格 : <= ; 严格 : < )
    return sgn( (b.t - b.s) / (a.s - b.s) ) * sgn( (b.t - b.s) / (a.t - b.s) ) <= 0;
}
P xLL(L a, L b) {  // 直线 a 和直线 b 的交点
    T s1 = (b.t - b.s) / (a.s - b.s), s2 = (a.t - b.s) / (b.t - b.s);
    return (a.s * s2 + a.t * s1) / (s1 + s2);
}
db disPL(P p, L a) { return fabs( (a.t-a.s) / (p-a.s) ) / (a.t-a.s).abs(); }  // 点 p 到直线 a 的距离
db disPS(P p, L a) {                                                          // 点 p 到线段 a 的距离
    return sgn( (a.t-a.s) * (p-a.s) ) * sgn( (a.s-a.t) * (p-a.t) ) == 1
           ? disPL(p, a) : min( (p-a.s).abs(), (p-a.t).abs() );
}
db disSS(L a, L b){                                                           // 线段 a 到线段 b 的距离
    if (xSS(a,b)) return 0;
    return min( min( disPS(a.s,b), disPS(a.t,b) ), min( disPS(b.s,a), disPS(b.t,a) ) );
}
db disLL(L a, L b) { return (a.t-a.s) / (b.t-b.s) ? 0 : disPL(a.s, b); }      // 直线 a 到直线 b 的距离

// 多边形、凸包
typedef vector<P> polygon;
polygon Convex(polygon A) {  // 求凸包，逆时针排序（ 内角 <180 : (<=) ; 内角 <=180 : (<) ）
    int n = sz(A); if (n <= 1) return A;
    sort(all(A));
    polygon B;
    for (int i = 0; i < n; B.pb(A[i++]))
        while (sz(B) > 1 && crossSgn(B[sz(B)-2], B.back(), A[i]) < 0) B.pop_back();  // 更改这里的 <
    for (int i = n - 2, t = sz(B); i >= 0; B.pb(A[i--]))
        while (sz(B) > t && crossSgn(B[sz(B)-2], B.back(), A[i]) < 0) B.pop_back();  // 同上
    B.pop_back(); return B;
}
