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
