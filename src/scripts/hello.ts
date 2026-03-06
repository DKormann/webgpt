console.log("HELLO WORLD")



type I <T> = (x:T) => {data:number}

type B <A, B> = (a:I<A> , b:I<B>) => I<A&B>



const x : I<{e:number}> = ({e}) => ({data:e})

const add : B<{e:number}, {f:number}> = (a,b)=> (k)=> ({data: a(k).data + b(k).data})

const apply = <T> (t:I<T>, k:string, val: number) : I<Omit<T,k>> => (t,k,val)=> ({data: t(k)

type D =  Omit<{e: number, f:number}, "e">

