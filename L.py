
class CollBase:
    "Base class for composing a list of `items`"
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, k): return self.items[list(k) if isinstance(k,CollBase) else k]
    def __setitem__(self, k, v): self.items[list(k) if isinstance(k,CollBase) else k] = v
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self): return self.items.__repr__()
    def __iter__(self): return self.items.__iter__()
    
class _L_Meta(type):
    def __call__(cls, x=None, *args, **kwargs):
        if not args and not kwargs and x is not None and isinstance(x,cls): return x
        return super().__call__(x, *args, **kwargs)

class L(GetAttr, CollBase, metaclass=_L_Meta):
    "Behaves like a list of `items` but can also index with list of indices or masks"
    _default='items'
    def __init__(self, items=None, *rest, use_list=False, match=None):
        if (use_list is not None) or not is_array(items):
            items = listify(items, *rest, use_list=use_list, match=match)
        super().__init__(items)

    @property
    def _xtra(self): return None
    def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
    def __getitem__(self, idx): return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
    def copy(self): return self._new(self.items.copy())

    def _get(self, i):
        if is_indexer(i) or isinstance(i,slice): return getattr(self.items,'iloc',self.items)[i]
        i = mask2idxs(i)
        return (self.items.iloc[list(i)] if hasattr(self.items,'iloc')
                else self.items.__array__()[(i,)] if hasattr(self.items,'__array__')
                else [self.items[i_] for i_ in i])

    def __setitem__(self, idx, o):
        "Set `idx` (can be list of indices, or mask, or int) items to `o` (which is broadcast if not iterable)"
        if isinstance(idx, int): self.items[idx] = o
        else:
            idx = idx if isinstance(idx,L) else listify(idx)
            if not is_iter(o): o = [o]*len(idx)
            for i,o_ in zip(idx,o): self.items[i] = o_

    def __eq__(self,b):
        if b is None: return False
        if risinstance('ndarray', b): return array_equal(b, self)
        if isinstance(b, (str,dict)): return False
        return all_equal(b,self)

    def sorted(self, key=None, reverse=False): return self._new(sorted_ex(self, key=key, reverse=reverse))
    def __iter__(self): return iter(self.items.itertuples() if hasattr(self.items,'iloc') else self.items)
    def __contains__(self,b): return b in self.items
    def __reversed__(self): return self._new(reversed(self.items))
    def __invert__(self): return self._new(not i for i in self)
    def __repr__(self): return repr(self.items)
    def _repr_pretty_(self, p, cycle):
        p.text('...' if cycle else repr(self.items) if is_array(self.items) else coll_repr(self))
    def __mul__ (a,b): return a._new(a.items*b)
    def __add__ (a,b): return a._new(a.items+listify(b))
    def __radd__(a,b): return a._new(b)+a
    def __addi__(a,b):
        a.items += list(b)
        return a

    @classmethod
    def split(cls, s, sep=None, maxsplit=-1): return cls(s.split(sep,maxsplit))
    @classmethod
    def range(cls, a, b=None, step=None): return cls(range_of(a, b=b, step=step))

    def map(self, f, *args, **kwargs): return self._new(map_ex(self, f, *args, gen=False, **kwargs))
    def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))
    def argfirst(self, f, negate=False): 
        if negate: f = not_(f)
        return first(i for i,o in self.enumerate() if f(o))
    def filter(self, f=noop, negate=False, **kwargs):
        return self._new(filter_ex(self, f=f, negate=negate, gen=False, **kwargs))

    def enumerate(self): return L(enumerate(self))
    def renumerate(self): return L(renumerate(self))
    def unique(self, sort=False, bidir=False, start=None): return L(uniqueify(self, sort=sort, bidir=bidir, start=start))
    def val2idx(self): return val2idx(self)
    def cycle(self): return cycle(self)
    def map_dict(self, f=noop, *args, **kwargs): return {k:f(k, *args,**kwargs) for k in self}
    def map_first(self, f=noop, g=noop, *args, **kwargs):
        return first(self.map(f, *args, **kwargs), g)

    def itemgot(self, *idxs):
        x = self
        for idx in idxs: x = x.map(itemgetter(idx))
        return x
    def attrgot(self, k, default=None):
        return self.map(lambda o: o.get(k,default) if isinstance(o, dict) else nested_attr(o,k,default))

    def starmap(self, f, *args, **kwargs): return self._new(itertools.starmap(partial(f,*args,**kwargs), self))
    def zip(self, cycled=False): return self._new((zip_cycle if cycled else zip)(*self))
    def zipwith(self, *rest, cycled=False): return self._new([self, *rest]).zip(cycled=cycled)
    def map_zip(self, f, *args, cycled=False, **kwargs): return self.zip(cycled=cycled).starmap(f, *args, **kwargs)
    def map_zipwith(self, f, *rest, cycled=False, **kwargs): return self.zipwith(*rest, cycled=cycled).starmap(f, **kwargs)
    def shuffle(self):
        it = copy(self.items)
        random.shuffle(it)
        return self._new(it)

    def concat(self): return self._new(itertools.chain.from_iterable(self.map(L)))
    def reduce(self, f, initial=None): return reduce(f, self) if initial is None else reduce(f, self, initial)
    def sum(self): return self.reduce(operator.add, 0)
    def product(self): return self.reduce(operator.mul, 1)
    def setattrs(self, attr, val): [setattr(o,attr,val) for o in self]
