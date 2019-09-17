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



//Splay伸展树
/*********************/
struct node { 
	#define MAXN 1000000
	int ch[MAXN][2],f[MAXN],size[MAXN],cnt[MAXN],key[MAXN];
	int sz,root;
	inline void clear(int x){
		ch[x][0]=ch[x][1]=f[x]=size[x]=cnt[x]=key[x]=0;
	}
	inline bool get(int x){
		return ch[f[x]][1]==x;
	}
	inline void update(int x){
		if (x){
			size[x]=cnt[x];
			if (ch[x][0]) size[x]+=size[ch[x][0]];
			if (ch[x][1]) size[x]+=size[ch[x][1]];
		}
	}
	inline void rotate(int x){
		int old=f[x],oldf=f[old],whichx=get(x);
		ch[old][whichx]=ch[x][whichx^1]; f[ch[old][whichx]]=old;
		ch[x][whichx^1]=old; f[old]=x;
		f[x]=oldf;
		if (oldf)
			ch[oldf][ch[oldf][1]==old]=x;
		update(old); update(x);
	}
	inline void splay(int x){
		for (int fa;fa=f[x];rotate(x))
		  if (f[fa])
		    rotate((get(x)==get(fa))?fa:x);
		root=x;
	}
	inline void insert(int x){
		if (root==0){sz++; ch[sz][0]=ch[sz][1]=f[sz]=0; root=sz; size[sz]=cnt[sz]=1; key[sz]=x; return;}
		int now=root,fa=0;
		while(1){
			if (x==key[now]){
				cnt[now]++; update(now); update(fa); splay(now); break;
			}
			fa=now;
			now=ch[now][key[now]<x];
			if (now==0){
				sz++;
				ch[sz][0]=ch[sz][1]=0;
				f[sz]=fa;
				size[sz]=cnt[sz]=1;
				ch[fa][key[fa]<x]=sz;
				key[sz]=x;
				update(fa);
				splay(sz);
				break;
			}
		}
	}
	inline int find(int x){
		int now=root,ans=0;
		while(1){
			if (x<key[now])
			  now=ch[now][0];
			else{
				ans+=(ch[now][0]?size[ch[now][0]]:0);
				if (x==key[now]){
					splay(now); return ans+1;
				}
				ans+=cnt[now];
				now=ch[now][1];
			}
		}
	}
	inline int findx(int x){
		int now=root;
		while(1){
			if (ch[now][0]&&x<=size[ch[now][0]])
			  now=ch[now][0];
			else{
				int temp=(ch[now][0]?size[ch[now][0]]:0)+cnt[now];
				if (x<=temp) return key[now];
				x-=temp; now=ch[now][1];
			}
		}
	}
	inline int pre(){
		int now=ch[root][0];
		while (ch[now][1]) now=ch[now][1];
		return now;
	}
	inline int next(){
		int now=ch[root][1];
		while (ch[now][0]) now=ch[now][0];
		return now;
	}
	inline void del(int x){
		int whatever=find(x);
		if (cnt[root]>1){cnt[root]--; update(root); return;}
		if (!ch[root][0]&&!ch[root][1]) {clear(root); root=0; return;}
		if (!ch[root][0]){
			int oldroot=root; root=ch[root][1]; f[root]=0; clear(oldroot); return;
		}
		else if (!ch[root][1]){
			int oldroot=root; root=ch[root][0]; f[root]=0; clear(oldroot); return;
		}
		int leftbig=pre(),oldroot=root;
		splay(leftbig);
		ch[root][1]=ch[oldroot][1];
		f[ch[oldroot][1]]=root;
		clear(oldroot);
		update(root); 
	}
}SPL;
int main(){
	int n,opt,x;
	scanf("%d",&n);
	for (int i=1;i<=n;++i){
		scanf("%d%d",&opt,&x);
		switch(opt){
			case 1: SPL.insert(x); break;
			case 2: SPL.del(x); break;
			case 3: printf("%d\n",SPL.find(x)); break;
			case 4: printf("%d\n",SPL.findx(x)); break;
			case 5: SPL.insert(x); printf("%d\n",SPL.key[SPL.pre()]); SPL.del(x); break;
			case 6: SPL.insert(x); printf("%d\n",SPL.key[SPL.next()]); SPL.del(x); break;
		}
	}
	/* 您需要写一种数据结构（可参考题目标题），来维护一些数，其中需要提供以下操作：
	1. 插入x数
	2. 删除x数(若有多个相同的数，因只删除一个)
	3. 查询x数的排名(若有多个相同的数，因输出最小的排名)
	4. 查询排名为x的数
	5. 求x的前驱(前驱定义为小于x，且最大的数)
	6. 求x的后继(后继定义为大于x，且最小的数)
	n个节点m次操作splay
	O(nlog2(n)+mlog2(n))*/
}
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
