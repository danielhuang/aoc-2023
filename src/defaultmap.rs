use std::borrow::Borrow;
use std::collections::hash_map::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::iter::{FromIterator, IntoIterator};
use std::ops::{Index, IndexMut};

/// A `HashMap` that returns a default when keys are accessed that are not present.
#[derive(PartialEq, Eq, Clone, Debug)]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DefaultHashMap<K: Eq + Hash, V: Clone> {
    map: HashMap<K, V>,
    pub default: V,
}

impl<K: Eq + Hash, V: Default + Clone> Default for DefaultHashMap<K, V> {
    /// The `default()` constructor creates an empty DefaultHashMap with the default of `V`
    /// as the default for missing keys.
    /// This is desired default for most use cases, if your case requires a
    /// different default you should use the `new()` constructor.
    fn default() -> DefaultHashMap<K, V> {
        DefaultHashMap {
            map: HashMap::default(),
            default: V::default(),
        }
    }
}

impl<K: Eq + Hash, V: Default + Clone> From<HashMap<K, V>> for DefaultHashMap<K, V> {
    /// If you already have a `HashMap` that you would like to convert to a
    /// `DefaultHashMap` you can use the `into()` method on the `HashMap` or the
    /// `from()` constructor of `DefaultHashMap`.
    /// The default value for missing keys will be `V::default()`,
    /// if this is not desired `DefaultHashMap::new_with_map()` should be used.
    fn from(map: HashMap<K, V>) -> DefaultHashMap<K, V> {
        DefaultHashMap {
            map,
            default: V::default(),
        }
    }
}

impl<K: Eq + Hash, V: Clone> Into<HashMap<K, V>> for DefaultHashMap<K, V> {
    /// The into method can be used to convert a `DefaultHashMap` back into a
    /// `HashMap`.
    fn into(self) -> HashMap<K, V> {
        self.map
    }
}

impl<K: Eq + Hash, V: Clone> DefaultHashMap<K, V> {
    /// Creates an empty `DefaultHashMap` with `default` as the default for missing keys.
    /// When the provided `default` is equivalent to `V::default()` it is preferred to use
    /// `DefaultHashMap::default()` instead.
    pub fn new(default: V) -> DefaultHashMap<K, V> {
        DefaultHashMap {
            map: HashMap::new(),
            default,
        }
    }

    /// Creates a `DefaultHashMap` based on a default and an already existing `HashMap`.
    /// If `V::default()` is the supplied default, usage of the `from()` constructor or the
    /// `into()` method on the original `HashMap` is preferred.
    pub fn new_with_map(default: V, map: HashMap<K, V>) -> DefaultHashMap<K, V> {
        DefaultHashMap { map, default }
    }

    /// Changes the default value permanently or until `set_default()` is called again.
    pub fn set_default(&mut self, new_default: V) {
        self.default = new_default;
    }

    /// Returns a reference to the value stored for the provided key.
    /// If the key is not in the `DefaultHashMap` a reference to the default value is returned.
    /// Usually the `map[key]` method of retrieving keys is preferred over using `get` directly.
    /// This method accepts both references and owned values as the key.
    pub fn get<Q, QB: Borrow<Q>>(&self, key: QB) -> &V
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.map.get(key.borrow()).unwrap_or(&self.default)
    }

    /// Returns a mutable reference to the value stored for the provided key.
    /// If there is no value stored for the key the default value is first inserted for this
    /// key before returning the reference.
    /// Usually the `map[key] = new_val` is prefered over using `get_mut` directly.
    /// This method only accepts owned values as the key.
    pub fn get_mut(&mut self, key: K) -> &mut V {
        let default: &V = &self.default;
        self.map.entry(key).or_insert_with(|| default.clone())
    }
}

/// Implements the `Index` trait so you can do `map[key]`.
/// Nonmutable indexing can be done both by passing a reference or an owned value as the key.
impl<'a, K: Eq + Hash, KB: Borrow<K>, V: Clone> Index<KB> for DefaultHashMap<K, V> {
    type Output = V;

    fn index(&self, index: KB) -> &V {
        self.get(index)
    }
}

/// Implements the `IndexMut` trait so you can do `map[key] = val`.
/// Mutably indexing can only be done when passing an owned value as the key.
impl<K: Eq + Hash, V: Clone> IndexMut<K> for DefaultHashMap<K, V> {
    #[inline]
    fn index_mut(&mut self, index: K) -> &mut V {
        self.get_mut(index)
    }
}

/// These methods simply forward to the underlying `HashMap`, see that
/// [documentation](https://doc.rust-lang.org/std/collections/struct.HashMap.html) for
/// the usage of these methods.
impl<K: Eq + Hash, V: Clone> DefaultHashMap<K, V> {
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional)
    }
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit()
    }
    #[inline]
    pub fn keys(&self) -> Keys<K, V> {
        self.map.keys()
    }
    #[inline]
    pub fn values(&self) -> Values<K, V> {
        self.map.values()
    }
    #[inline]
    pub fn values_mut(&mut self) -> ValuesMut<K, V> {
        self.map.values_mut()
    }
    #[inline]
    pub fn iter(&self) -> Iter<K, V> {
        self.map.iter()
    }
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        self.map.iter_mut()
    }
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        self.map.entry(key)
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    #[inline]
    pub fn drain(&mut self) -> Drain<K, V> {
        self.map.drain()
    }
    #[inline]
    pub fn clear(&mut self) {
        self.map.clear()
    }
    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.map.insert(k, v)
    }
    #[inline]
    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.map.contains_key(k)
    }
    #[inline]
    pub fn remove<Q>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.map.remove(k)
    }
    #[inline]
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.map.retain(f)
    }
}

impl<K: Eq + Hash, V: Default + Clone> FromIterator<(K, V)> for DefaultHashMap<K, V> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
    {
        Self {
            map: HashMap::from_iter(iter),
            default: V::default(),
        }
    }
}
