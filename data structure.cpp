//树状数组 
/*********************/
ll C[N];
int n;
ll lowbit(ll t) { return t&(-t); }
ll getsum(ll x) {
	ll ans=0;
	for(ll i=x;i>0;i-=lowbit(i))
		ans+=C[i];
	return ans;
}
void add(ll x,ll y) { for(ll i=x;i<=n;i+=lowbit(i)) C[i]+=y; }
/*********************/


//树状数组区间修改区间查询
/*********************/
int n,m;
const int N=1e5+5;
ll sum1[N],sum2[N];
ll lowbit(ll n) { return (n&(-n)); }
ll add(ll p,ll x) {
	for(int i=p;i<=n;i+=lowbit(i)) 
		sum1[i]+=x,sum2[i]+=x*p; }
ll range_add(ll l,ll r,ll x) {
	add(l,x),add(r+1,-x); }
ll ask(ll p) {
	ll res=0;
	for(int i=p;i;i-=lowbit(i)) {
		res+=(p+1)*sum1[i]-sum2[i];
	}
	return res; }
ll range_ask(ll l,ll r) {
	return (ask(r)-ask(l-1)); }
int main() {
	scanf("%lld%lld",&n,&m);
	ll last=0;
	for(int i=1;i<=n;i++) {
		ll now;
		scanf("%lld",&now);
		add(i,now-last);
		last=now;
	}
	for(int i=1;i<=m;i++) {
		ll op,L,R;
		scanf("%lld%lld%lld",&op,&L,&R);
		if(op==1) {
			ll x;
			scanf("%lld",&x);
			range_add(L,R,x);
		}else {
			printf("%lld\n",range_ask(L,R));
		}
	}
}
/*********************/



//线段树
/*********************/
const int N=3e5+5;
const int MAXN=3e5+5;
ll a[MAXN];
struct STree{
	#define ls p<<1 
	#define rs p<<1|1
	#define lls l,mid,ls
	#define rrs mid+1,r,rs
	#define qm (l+r)>>1
	ll ans[MAXN<<2],tag[MAXN<<2];
	inline void push_up(ll p) { ans[p]=min(ans[ls],ans[rs]); }
	void build(ll l,ll r,ll p) {
	    tag[p]=0;
	    if(l==r){ans[p]=a[l];return ;}
	    ll mid=qm;
	    build(lls); build(rrs);
	    push_up(p); }
	inline void up(ll l,ll r,ll p,ll k) {
	    tag[p]=tag[p]+k;
	    ans[p]=ans[p]+k*(r-l+1); }
	inline void push_down(ll p,ll l,ll r) {
	    ll mid=qm;
	    up(lls,tag[p]); up(rrs,tag[p]);
	    tag[p]=0; }
	inline void update(ll nl,ll nr,ll l,ll r,ll p,ll k) {
	    if(nl<=l&&r<=nr) {
	        ans[p]+=k*(r-l+1);
	        tag[p]+=k;
	        return ; }
	    push_down(p,l,r);
	    ll mid=qm;
	    if(nl<=mid)update(nl,nr,lls,k);
		if(nr>mid) update(nl,nr,rrs,k);
	    push_up(p); }
	ll query(ll q_x,ll q_y,ll l,ll r,ll p) {
	    ll res=INF;
	    if(q_x<=l&&r<=q_y)return ans[p];
	    ll mid=qm;
	    push_down(p,l,r);
	    if(q_x<=mid)res=min(res,query(q_x,q_y,lls));
		if(q_y>mid) res=min(res,query(q_x,q_y,rrs));
	    return res; }
}ST;
/*********************/


//主席树 
/*********************/
ll tot;
int getid(int x) {return lower_bound(v.begin(),v.end(),x)-v.begin()+1;}
struct STZXTree{
	#define qm (l+r)>>1 
	struct node{ int l,r,sum; } T[N*20];
	void build(int l,int r,int &x){
	    x=++tot;
	    T[x].sum=0;
	    if(l==r)return;
	    int m=(l+r)>>1;
	    build(l,m,T[x].l);
	    build(m+1,r,T[x].r);}
	void updata(int l,int r,int &x,int y,int pos) {
		de(tot);
		tot++;
		T[tot]=T[y];T[tot].sum++;x=tot;
		if(l==r) return ;
		int mid=qm;
		if(pos<=mid) updata(l,mid,T[x].l,T[y].l,pos);
		else updata(mid+1,r,T[x].r,T[y].r,pos);}
	int query(int l,int r,int x,int y,int k) {
		if(l==r) return l;
		int sum=T[T[y].l].sum-T[T[x].l].sum;
		int mid=qm;
		if(k<=sum) return query(l,mid,T[x].l,T[y].l,k);
		else return query(mid+1,r,T[x].r,T[y].r,k-sum);}
}SST;


/*cout << v[SST.query(1,hh,root[l1-1],root[r1],kk)-1] << endl;
int hh=v.size();tot=0;SST.build(1,hh,root[0]);
ff(i,1,n) SST.updata(1,hh,root[i],root[i-1],getid(a[i]));*/
/*********************/




//区间某k大之间求个数
/*********************/
const int N=2e5+5;

vector<int> v;
int a[N<<2];
int rt[N<<2];
ll tot;
int d;
int getid(int x) {return lower_bound(v.begin(),v.end(),x)-v.begin()+1;}
int getid2(int x) {return upper_bound(v.begin(),v.end(),x)-v.begin()+1;}
struct STZXTree{
    #define qm (l+r)>>1 
    struct node{ int l,r,sum; } T[N*40];
    void build(int l,int r,int &x){
        x=++tot;
        T[x].sum=0;
        if(l==r)return;
        int m=(l+r)>>1;
        build(l,m,T[x].l);
        build(m+1,r,T[x].r);}
    void updata(int l,int r,int &x,int y,int pos) {
        tot++;
        T[tot]=T[y];
        T[tot].sum++;
        x=tot;
        if(l==r) return ;
        int mid=qm;
        if(pos<=mid) updata(l,mid,T[x].l,T[y].l,pos);
        else updata(mid+1,r,T[x].r,T[y].r,pos);
    }
    
    ll query(ll l,ll r,ll x,ll y,ll k) {
        if(l==r) return l;
        ll sum=T[T[y].l].sum-T[T[x].l].sum;
        ll mid=qm;
        if(k<=sum) return query(l,mid,T[x].l,T[y].l,k);
        else return query(mid+1,r,T[x].r,T[y].r,k-sum);
    }    
    
    void qry(int rt, int sub, int L, int R, int l, int r) {
        if(rt==0&&sub==0) return ;
        if(L <= l && r <= R) {
            if(rt!=0)
            d+=(T[rt].sum - T[sub].sum);
            return ;
        }
        int mid = l + r >> 1;
        if(L <= mid) qry(T[rt].l, T[sub].l,  L, R, l, mid);
        if(R > mid) qry(T[rt].r, T[sub].r, L, R, mid + 1, r);
    }
}SST;
int l1,r1,k,hh,p;
bool check(int x){
    d=0;
    int l2=getid(p-x);
    int r2=getid2(p+x)-1;
      SST.qry( rt[r1], rt[l1-1], l2, r2, 1, hh ); 
    if(d>=k)return true;
    return false;
}
/*********************/



//lca 预处理nlog(n)  查询log(n)
/*********************************************/
const int MAXN=500005;
int rmq[2*MAXN];
struct ST {
	int mm[2*MAXN];
	int dp[2*MAXN][20];
	void init(int n) {
		mm[0]=-1;
		ff(i,1,n) {
			mm[i]=((i&(i-1))==0)?mm[i-1]+1:mm[i-1];
			dp[i][0]=i;
		}
		ff(j,1,mm[n])
			for(int i=1;i+(1<<j)-1<=n;i++)
				dp[i][j]=rmq[dp[i][j-1]] <
	rmq[dp[i+(1<<(j-1))][j-1]]?dp[i][j-1]:dp[i+(1<<(j-1))][j-1];
	}
	int query(int a,int b) {
		if(a>b) swap(a,b);
		int k=mm[b-a+1];
		return rmq[dp[a][k]]<=
	rmq[dp[b-(1<<k)+1][k]]?dp[a][k]:dp[b-(1<<k)+1][k];
	}
};
struct Edge { int to,next;};
Edge edge[MAXN*2];
int tot,head[MAXN];
int F[MAXN*2];
int P[MAXN];
int cnt;
ST st;
void init() {
	tot=0;
	mem(head,-1);
}
void addedge(int u,int v) {
	edge[tot].to=v;
	edge[tot].next=head[u];
	head[u]=tot++;
}
void dfs(int u,int pre,int dep) {
	F[++cnt]=u;
	rmq[cnt]=dep;
	P[u]=cnt;
	for(int i=head[u];i!=-1;i=edge[i].next) {
		int v=edge[i].to;
		if(v==pre)continue;
		dfs(v,u,dep+1);
		F[++cnt]=u;
		rmq[cnt]=dep;
	}
}
void LCA_init(int root,int node_num) {
	cnt=0;
	dfs(root,root,0);
	st.init(2*node_num-1);
}
int query_lca(int u,int v) {
	return F[st.query(P[u],P[v])];
}
bool flag[MAXN];
int main() {
	int T=1;
	int n;
	int u,v;
	//scanf("%d",&T);
	ff(tt,1,T) {
		int root;
		int k;
		scanf("%d%d%d",&n,&k,&root);
		init();
		mem(flag,0);
		ff(i,1,n-1) {
			scanf("%d%d",&u,&v);
			addedge(u,v);
			addedge(v,u);
		}
		/* 
		ff(i,1,n) if(!flag[i]) {
				root=i;
				break;
		}*/ 
		LCA_init(root,n);
		
		//printf("Case %d:\n",tt);
		ff(i,1,k) {
			scanf("%d%d",&u,&v);
		//	dd(u);de(v);
			printf("%d\n",query_lca(u,v));
		}
	}
}
/*********************************************/



//Tarjan算法 复杂度O(n+Q)
/***************************/
const int MAXN=5e5+5;
const int MAXQ=5e5+5;
int F[MAXN];
int find(int x) {
	if(F[x]==-1) return x;
	return F[x]=find(F[x]);
}
void bing(int u,int v) {
	int t1=find(u);
	int t2=find(v);
	if(t1!=t2) F[t1]=t2;
}

bool vis[MAXN];
int ancestor[MAXN];
struct Edge{
	int to,next;
}edge[MAXN*2];
int head[MAXN],tot;
void addedge(int u,int v) {
	edge[tot].to=v;
	edge[tot].next=head[u];
	head[u]=tot++;
}
struct Query {
	int q,next;
	int index;
}query[MAXQ*2];
int answer[MAXQ];
int h[MAXQ];
int tt;
int Q;
void add_query(int u,int v,int index) {
	query[tt].q=v;
	query[tt].next=h[u];
	query[tt].index=index;
	h[u]=tt++;
	query[tt].q=u;
	query[tt].next=h[v];
	query[tt].index=index;
	h[v]=tt++;
}
void init() {
	tot=0;
	mem(head,-1);
	tt=0;
	mem(h,-1);
	mem(vis,0);
	mem(F,-1);
	mem(ancestor,0);
}
void LCA(int u) {
	ancestor[u]=u;
	vis[u]=true;
	for(int i=head[u];i!=-1;i=edge[i].next) {
		int v=edge[i].to;
		if(vis[v])continue;
		LCA(v);
		bing(u,v);
		ancestor[find(u)]=u;
	}
	for(int i=h[u];i!=-1;i=query[i].next) {
		int v=query[i].q;
		if(vis[v]) {
			answer[query[i].index]=ancestor[find(v)];
		}
	}
}
bool flag[MAXN];
int main() {
	int n;
	int u,v,k;
	int t;
	//scanf("%d",&t);
	t=1;
	ff(ii,1,t) {
		int root;
		int k;
		//printf("Case %d:\n",ii);
		init();
		mem(flag,0);
		scanf("%d%d%d",&n,&k,&root);
		ff(i,1,n-1) {
			scanf("%d%d",&u,&v);
			addedge(u,v);
			addedge(v,u);
		}
		ff(i,1,k) {
			scanf("%d%d",&u,&v);
		//	dd(u);de(v);
			add_query(u,v,i);
		}
		/*
		ff(i,1,n) if(!flag[i]) {
			root=i;
			break;
		}*/
		LCA(root);
		ff(i,1,k) printf("%d\n",answer[i]);
	}	
}
/***************************/






lca倍增
/***************************/
const int MAXN=5e5+5;
const int DEG=20;
struct Edge {
	int to,next;
}edge[MAXN*2];
int head[MAXN],tot;
void addedge(int u,int v) {
	edge[tot].to=v;
	edge[tot].next=head[u];
	head[u]=tot++;
}
void init() {
	tot=0;
	mem(head,-1);
}
int fa[MAXN][DEG];
int deg[MAXN];
void BFS(int root) {
	queue<int>que;
	deg[root]=0;
	fa[root][0]=root;
	que.push(root);
	while(!que.empty()) {
		int tmp=que.front();
		que.pop();
		ff(i,1,DEG-1)
			fa[tmp][i]=fa[fa[tmp][i-1]][i-1];
		for(int i=head[tmp];i!=-1;i=edge[i].next) {
			int v=edge[i].to;
			if(v==fa[tmp][0]) continue;
			deg[v]=deg[tmp]+1;
			fa[v][0]=tmp;
			que.push(v);
		}
	}
}
int LCA(int u,int v) {
	if(deg[u]>deg[v]) swap(u,v);
	int hu=deg[u],hv=deg[v];
	int tu=u,tv=v;
	for(int det=hv-hu,i=0;det;det>>=1,i++)
		if(det&1)
			tv=fa[tv][i];
	if(tu==tv) return tu;
	fd(i,DEG-1,0) {
	//for(int i=DEG-1;i>=0;i--){
		if(fa[tu][i]==fa[tv][i]) continue;
		tu=fa[tu][i];
		tv=fa[tv][i];
	}
	return fa[tu][0];
}
bool flag[MAXN];
int main() {
	int T;
	T=1; 
	ff(tt,1,T) {
		init();
		mem(flag,0);
		int n;
		int k;;
		int root;
		scanf("%d%d%d",&n,&k,&root);
	//	de(root);
		ff(i,1,n-1) {
			int u,v;
			scanf("%d%d",&u,&v);
			addedge(u,v);
			addedge(v,u);
			//flag[v]=i;
		}
			/*int root;
			ff(i,1,n) if(!flag[i]) {
				root=i;
				break;
			}*/
		//	de(root);
			BFS(root);
			ff(i,1,k) {
				int u,v;
				scanf("%d%d",&u,&v);
				//(u);de(v);
				printf("%d\n",LCA(u,v));
			}
	}
}
/****************************/


树链剖分
操作1： 格式： 1 x y z 表示将树从x到y结点最短路径上所有节点的值都加上z
操作2： 格式： 2 x y 表示求树从x到y结点最短路径上所有节点的值之和
操作3： 格式： 3 x z 表示将以x为根节点的子树内所有节点值都加上z
操作4： 格式： 4 x 表示求以x为根节点的子树内所有节点值之和
/***************************/
#include <bits/stdc++.h>
using namespace std;
#define mem(a, b) memset(a, b, sizeof(a))
#define lson l, m, rt << 1
#define rson m + 1, r, rt << 1 | 1
const int N = 2e5 + 10;

#define de(a) cout << #a << " = " << a << endl;
#define dd(a) cout << #a << " = " << a << "  ";
int sum[N << 2], lazy[N << 2]; //线段树求和
int n, m, r, mod;              //节点数，操作数，根节点，模数
int first[N], tot;             //邻接表
//分别为:重儿子，每个节点新编号，父亲，编号，深度，子树个数，所在重链的顶部
int son[N], id[N], fa[N], cnt, dep[N], siz[N], top[N];
int w[N], wt[N]; //初始点权,新编号点权
int res = 0;     //查询答案
struct edge
{
    int v, next;
} e[N];
void add_edge(int u, int v)
{
    e[tot].v = v;
    e[tot].next = first[u];
    first[u] = tot++;
}
void init()
{
    mem(first, -1);
    tot = 0;
    cnt = 0;
}
int pushup(int rt)
{
    sum[rt] = (sum[rt << 1] + sum[rt << 1 | 1]) % mod;
}
void pushdown(int rt, int m) //下放lazy标记
{
    if (lazy[rt])
    {
        lazy[rt << 1] += lazy[rt];                 //给左儿子下放lazy
        lazy[rt << 1 | 1] += lazy[rt];             //给右儿子下放lazy
        sum[rt << 1] += lazy[rt] * (m - (m >> 1)); //更新sum
        sum[rt << 1] %= mod;
        sum[rt << 1 | 1] += lazy[rt] * (m >> 1);  
        sum[rt << 1 | 1] %= mod;
        lazy[rt] = 0;
    }
}
void build(int l, int r, int rt)
{
    lazy[rt] = 0;
    if (l == r)
    {
        sum[rt] = wt[l]; //新的编号点权
        sum[rt] %= mod;
        return;
    }
    int m = (l + r) >> 1;
    build(lson);
    build(rson);
    pushup(rt);
}
void update(int L, int R, int c, int l, int r, int rt)
{
    if (L <= l && r <= R)
    {
        lazy[rt] += c;
        sum[rt] += c * (r - l + 1);
        sum[rt] %= mod;
        return;
    }
    pushdown(rt, r - l + 1);
    int m = (l + r) >> 1;
    if (L <= m)
        update(L, R, c, lson);
    if (R > m)
        update(L, R, c, rson);
    pushup(rt);
}
void query(int L, int R, int l, int r, int rt)
{
    if (L <= l && r <= R)
    {
        res += sum[rt];
        res %= mod;
        return;
    }
    pushdown(rt, r - l + 1);
    int m = (l + r) >> 1;
    if (L <= m)
        query(L, R, lson);
    if (R > m)
        query(L, R, rson);
}
//----------------------------------------------------------------
//处理出fa[],dep[],siz[],son[]
void dfs1(int u, int f, int deep)
{
    dep[u] = deep;   //标记深度
    fa[u] = f;       //标记节点的父亲
    siz[u] = 1;      //记录每个节点子树大小
    int maxson = -1; //记录重儿子数量
//    de(first[u]);
    for (int i = first[u]; ~i; i = e[i].next)
    {
    //	dd(i);
        int v = e[i].v;
        if (v == f)
            continue;
        dfs1(v, u, deep + 1);
        siz[u] += siz[v];
        if (siz[v] > maxson) //儿子里最多siz就是重儿子
        {
            son[u] = v; //标记u的重儿子为v
            maxson = siz[v];
        }
    }//cout << endl;
}

//处理出top[],wt[],id[]
void dfs2(int u, int topf)
{
    id[u] = ++cnt;  //每个节点的新编号
    wt[cnt] = w[u]; //新编号的对应权值
    top[u] = topf;  //标记每个重链的顶端
    if (!son[u])    //没有儿子时返回
        return;
    dfs2(son[u], topf); //搜索下一个重儿子
    for (int i = first[u]; ~i; i = e[i].next)
    {
        int v = e[i].v;
        if (v == fa[u] || v == son[u]) //处理轻儿子
            continue;
        dfs2(v, v); //每一个轻儿子都有一个从自己开始的链
    }
}
void updrange(int x, int y, int k)
{
    k %= mod;
//    dd(x);de(top[x]);
//    dd(y);de(top[y]);
    while (top[x] != top[y])
    {
        if (dep[top[x]] < dep[top[y]]) //使x深度较大
            swap(x, y);
//        dd(x);de(y);
//        dd(id[top[x]]);de(id[x]);
        update(id[top[x]], id[x], k, 1, n, 1);
        x = fa[top[x]];
    }
    if (dep[x] > dep[y]) //使x深度较小
        swap(x, y);
    update(id[x], id[y], k, 1, n, 1);
}
int qrange(int x, int y)
{
    int ans = 0;
    while (top[x] != top[y]) //当两个点不在同一条链上
    {
        if (dep[top[x]] < dep[top[y]]) //使x深度较大
            swap(x, y);
        res = 0;
        query(id[top[x]], id[x], 1, n, 1);
        //ans加上x点到x所在链顶端这一段区间的点权和
        ans += res;
        ans %= mod;
        x = fa[top[x]]; //x跳到x所在链顶端的这个点的上面一个点
    }
    //当两个点处于同一条链
    if (dep[x] > dep[y]) //使x深度较小
        swap(x, y);
    res = 0;
    query(id[x], id[y], 1, n, 1);
    ans += res;
    return ans % mod;
}
void upson(int x, int k)
{
    update(id[x], id[x] + siz[x] - 1, k, 1, n, 1); //子树区间右端点为id[x]+siz[x]-1
}
int qson(int x)
{
    res = 0;
    query(id[x], id[x] + siz[x] - 1, 1, n, 1);
    return res;
}
int main()
{
    // freopen("in.txt", "r", stdin);
    int u, v;
    scanf("%d%d%d%d", &n, &m, &r, &mod);
    init();
    for (int i = 1; i <= n; i++)
        scanf("%d", &w[i]);
    for (int i = 1; i <= n - 1; i++)
    {
        scanf("%d%d", &u, &v);
        add_edge(u, v);
        add_edge(v, u);
    }
//    for (int i = 0;i <= n+n; i++) cout << e[i].v << ' ';cout << endl; 
//    for (int i = 0;i <= n+n; i++) cout << e[i].next << ' ';cout << endl; 
//    for (int i = 0;i <= n+n; i++) cout << first[i] << ' ';cout << endl; 
    dfs1(r, 0, 1);
    dfs2(r, r);
    build(1, n, 1); //用新点权建立线段树
    while (m--)
    {
        int op, x, y, z;
        scanf("%d", &op);
        if (op == 1)
        {
            scanf("%d%d%d", &x, &y, &z);
            updrange(x, y, z);
        }
        else if (op == 2)
        {
            scanf("%d%d", &x, &y);
            printf("%d\n", qrange(x, y));
        }
        else if (op == 3)
        {
            scanf("%d%d", &x, &z);
            upson(x, z);
        }
        else if (op == 4)
        {
            scanf("%d", &x);
            printf("%d\n", qson(x));
        }
    }
    return 0;
}
/* 
5 5 2 24
7 3 7 8 0 
1 2
1 5
3 1
4 1
3 4 2
3 2 2
4 5
1 5 1 3
2 1 3
*/ 
/*************************************/



平衡树splay
/**************************************/
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<ll,ll>pll;
typedef double db;
/* Nothing is ture,  (22)
Everything is permitted*/
#define fi  first
#define se second
#define pb(x)        push_back(x)
#define mp(a,b)    make_pair(a,b)
#define sz(a)   ((int)(a).size())
#define all(x)  x.begin(),x.end()
#define msort(a) sort(a+1,a+1+n);
#define endl "\n"

#define ff(i,a,b) for(ll i=a;i<=b;++i)
#define fd(i,a,b) for(ll i=a;i>=b;--i)

#define de(a) cout << #a << " = " << a << endl;
#define dd(a) cout << #a << " = " << a << "  ";

#define INF 0x3f3f3f3f
#define INFF 0x3f3f3f3f3f3f3f3f
#define mem(a,b)   memset(a,b,sizeof(a))
#define file freopen("in.txt","r",stdin)
#define jkl ios::sync_with_stdio(false);cin.tie(0);
/*********************************************/
const int N=2e5+6;
int ch[N][2],par[N],val[N],cnt[N],size[N],ncnt,root;
bool chk(int x) {
	return ch[par[x]][1]==x;
}
void pushup(int x) {
	size[x]=size[ch[x][0]]+size[ch[x][1]]+cnt[x];
}
void rotate(int x) {
	int y=par[x],z=par[y],k=chk(x),w=ch[x][k^1];
	ch[y][k]=w;par[w]=y;
	ch[z][chk(y)]=x;par[x]=z;
	ch[x][k^1]=y;par[y]=x;
	pushup(y);pushup(x);
}
void splay(int x,int goal=0) {
	while(par[x]!=goal) {
		int y=par[x],z=par[y];
		if(z!=goal) {
			if(chk(x)==chk(y)) rotate(y);
			else rotate(x);
		}
		rotate(x);
	}
	if(!goal) root=x;
}
void insert(int x) {
	int cur=root,p=0;
	while(cur&&val[cur]!=x) {
		p=cur;
		cur=ch[cur][x>val[cur]];
	}
	if(cur) {
		cnt[cur]++;
	}else {
		cur=++ncnt;
		if(p)ch[p][x>val[p]]=cur;
		ch[cur][0]=ch[cur][1]=0;
		par[cur]=p;val[cur]=x;
		cnt[cur]=size[cur]=1;
	}
	splay(cur);
}
void find(int x) {
	int cur=root;
	while(ch[cur][x>val[cur]]&&x!=val[cur]) {
		cur=ch[cur][x>val[cur]];
	}
	splay(cur);
}
int kth(int k) {
	int cur=root;
	while(1) {
		if(ch[cur][0]&&k<=size[ch[cur][0]]) {
			cur=ch[cur][0];
		}else if(k>size[ch[cur][0]]+cnt[cur]) {
			k-=size[ch[cur][0]]+cnt[cur];
			cur=ch[cur][1];
		}else {
			return cur;
		}
	}
}
int pre(int x) {
	find(x);
	if(val[root]<x) return root;
	int cur=ch[root][0];
	while(ch[cur][1]) cur=ch[cur][1];
	return cur;
}
int succ(int x) {
	find(x);
	if(val[root]>x) return root;
	int cur=ch[root][1];
	while(ch[cur][0]) cur=ch[cur][0];
	return cur;
}
void remove(int x) {
	int last=pre(x),next=succ(x);
	splay(last);splay(next,last);
	int del=ch[next][0];
	if(cnt[del]>1) {
		cnt[del]--;
		splay(del);
	}
	else ch[next][0]=0;
}
int n,op,x;
int main() {
	scanf("%d",&n);
	insert(0x3f3f3f3f);
	insert(0xcfcfcfcf);
	while(n--) {
		scanf("%d%d",&op,&x);
		switch(op) {
			case 1: insert(x);break;
			case 2: remove(x);break;
			case 3: find(x);printf("%d\n",size[ch[root][0]]);break;
			case 4: printf("%d\n",val[kth(x+1)]);break;
			case 5: printf("%d\n",val[pre(x)]);break;
			case 6: printf("%d\n",val[succ(x)]);break;
		}
	}
}
/*****************************/
