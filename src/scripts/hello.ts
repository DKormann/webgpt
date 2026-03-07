console.log("HELLO WORLD")



type I <T> = (x:T) => {data:number}

type B <A, B> = (a:I<A> , b:I<B>) => I<A&B>

const x : I<{e:number}> = ({e}) => ({data:e})
const add : B<{e:number}, {f:number}> = (a,b)=> (k)=> ({data: a(k).data + b(k).data})


type Graph = {
  name: string
  input: (name:string) => Graph
  add: (a:Graph, b: Graph) => Graph
  execute: (args?: {[key:string]:Graph})=>number[]
  mk:(fn:(...x:Graph[])=>Graph) => Graph
}


const Graph : Graph  = {
  name: "Graph",
  input: (name:string) => ({...Graph, name}),
  add: (a:Graph,b:Graph) => ({...Graph,name: "add"}),
  execute: () => [],
  mk: fn => ({...Graph, ok:2})

}






Graph.mk(x=>Graph)