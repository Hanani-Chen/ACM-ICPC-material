int root[N];
struct STZXTree{
	#define qm (l+r)>>1 
	struct node{ int l,r,sum; } T[N*40];
	void updata(int l,int r,int &x,int y,int pos) {
		cnt++;
		T[cnt]=T[y];T[cnt].sum++;x=cnt;
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
/*SST.query(1,n,root[l1-1],root[r1],k)l1~rÇø¼äµÚk´ó*/
/*ff(i,1,n) SST.updata(1,n,root[i],root[i-1],getid(a[i]));*/
