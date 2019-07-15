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
