#![no_std]

//! A **heap-allocated**, **fixed-capacity**, **variable-size** array, `no_std` compatible.
//!
//! `CapVec<T, N>` is a heap-allocated container that stores up to `N` elements
//! of type `T` contiguously in memory.
//! Unlike [`Vec`], the capacity is **fixed at compile-time**, and cannot grow.
//! However, its length (`len`) can vary dynamically up to `N`.
//!
//! This type is ideal for cases where:
//! - You're working in `no_std` environments.
//! - You want a `Vec`-like API but need deterministic capacity.
//! - You prefer heap allocation over stack-based fixed arrays like `[T; N]`.
//!
//! ```rust
//! use cap_vec::CapVec;
//!
//! let mut v = CapVec::<i32, 4>::new();
//! assert!(v.is_empty());
//!
//! v.push(10).unwrap();
//! v.push(20).unwrap();
//! v.push(30).unwrap();
//! assert_eq!(v.len(), 3);
//!
//! assert_eq!(v.first(), Some(&10));
//! assert_eq!(v.get(1), Some(&20));
//! assert_eq!(v.last(), Some(&30));
//!
//! assert_eq!(&v[..], &[10, 20, 30]);
//! assert_eq!(v.remove(1), Some(20));
//! assert_eq!(v.pop(), Some(30));
//! assert_eq!(v.pop(), Some(10));
//! ```

extern crate alloc;

use alloc::boxed::Box;
use core::hash::Hash;
use core::iter::FusedIterator;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut, Range};
use core::slice::{Iter, IterMut};

/// A heap-allocated, fixed-capacity, variable-size array.
///
/// `CapVec` is similar to [`Vec`], except:
/// - The **capacity** is fixed and known at compile time.
/// - No reallocation or growth occurs once the capacity is reached.
/// - The internal buffer is allocated on the **heap** and initialized lazily.
#[derive(Default)]
pub struct CapVec<T, const N: usize> {
    len: usize,
    buf: Option<Box<[MaybeUninit<T>; N]>>,
}

impl<T, const N: usize> From<[T; N]> for CapVec<T, N> {
    fn from(value: [T; N]) -> Self {
        Self {
            len: value.len(),
            buf: Some(Box::new(value.map(MaybeUninit::new))),
        }
    }
}

impl<T, const N: usize> Clone for CapVec<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            len: self.len,
            buf: (!self.is_empty()).then(|| {
                let mut elements = Box::new([const { MaybeUninit::uninit() }; N]);
                elements
                    .iter_mut()
                    .zip(self.as_slice())
                    .for_each(|(dest, source)| {
                        dest.write(source.clone());
                    });

                elements
            }),
        }
    }
}

impl<T, const N: usize> CapVec<T, N> {
    /// Creates a new, empty `CapVec` with capacity `N`.
    /// The internal buffer is not allocated until an element is inserted.
    ///
    /// ```
    /// use cap_vec::CapVec;
    ///
    /// let mut v = CapVec::<i32, 8>::new();
    /// assert_eq!(v.capacity(), 8);
    /// assert_eq!(v.len(), 0);
    /// assert!(v.is_empty());
    /// ```
    pub const fn new() -> Self {
        Self { len: 0, buf: None }
    }

    /// Extends the `CapVec` with elements from an iterator, up to its capacity.
    /// Returns the remaining iterator once the vector is full or the iterator ends.
    ///
    /// ```
    /// use cap_vec::CapVec;
    ///
    /// let mut v = CapVec::<u8, 3>::new();
    /// let mut leftover = v.extend([1, 2, 3, 4]);
    ///
    /// assert_eq!(v.as_slice(), &[1, 2, 3]);
    /// assert_eq!(leftover.next(), Some(4)); // one element not consumed
    /// ```
    pub fn extend<I>(&mut self, iter: I) -> I::IntoIter
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();

        for _ in 0..N - self.len {
            match iter.next() {
                Some(element) => unsafe { self.push(element).unwrap_unchecked() },
                None => break,
            }
        }

        iter
    }

    /// Appends an element to the end of the collection.
    /// Returns `Ok(())` on success, or `Err(element)` if the vector is already full.
    ///
    /// ```
    /// use cap_vec::CapVec;
    ///
    /// let mut v = CapVec::<i32, 2>::new();
    /// v.push(10).unwrap();
    /// v.push(20).unwrap();
    /// assert_eq!(v.push(30), Err(30)); // full
    /// ```
    pub fn push(&mut self, element: T) -> Result<(), T> {
        if self.len >= N {
            return Err(element);
        }

        let ptr = self
            .buf
            .get_or_insert_with(|| Box::new([const { MaybeUninit::uninit() }; N]))
            .as_mut_ptr();

        unsafe {
            // Write the new element at the specified index
            ptr.add(self.len).write(MaybeUninit::new(element));
        }

        // Increment the size of the array
        self.len += 1;
        Ok(())
    }

    /// Inserts an element at the given index, shifting subsequent elements to the right.
    /// Returns `Err(element)` if the index is out of bounds or the vector is full.
    ///
    /// # Safety
    /// This function performs unchecked memory writes internally,
    /// but all unsafe operations are encapsulated safely.
    ///
    /// ```
    /// use cap_vec::CapVec;
    ///
    /// let mut v = CapVec::<i32, 3>::new();
    /// v.extend([1, 3]);
    /// v.insert(1, 2).unwrap();
    /// assert_eq!(v.as_slice(), &[1, 2, 3]);
    /// ```
    pub fn insert(&mut self, index: usize, element: T) -> Result<(), T> {
        if index > self.len || self.len >= N {
            return Err(element);
        }

        let ptr = self
            .buf
            .get_or_insert_with(|| Box::new([const { MaybeUninit::uninit() }; N]))
            .as_mut_ptr();

        unsafe {
            // Shift elements starting from the index to the right
            core::ptr::copy(ptr.add(index), ptr.add(index + 1), self.len - index);

            // Write the new element at the specified index
            ptr.add(index).write(MaybeUninit::new(element));
        }

        // Increment the size of the array
        self.len += 1;
        Ok(())
    }

    /// Removes the last element and returns it, or `None` if the vector is empty.
    ///
    /// ```
    /// use cap_vec::CapVec;
    ///
    /// let mut v = CapVec::<u8, 4>::new();
    /// v.extend([1, 2, 3]);
    /// assert_eq!(v.pop(), Some(3));
    /// assert_eq!(v.as_slice(), &[1, 2]);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        debug_assert!(self.buf.is_some());
        let buf = unsafe { self.buf.as_mut().unwrap_unchecked() };

        // Decrement the len of the array
        self.len -= 1;

        // Read the element to be removed
        let element = unsafe { buf[self.len].assume_init_read() };
        Some(element)
    }

    /// Removes and returns the element at the given index, shifting subsequent elements left.
    /// Returns `None` if the index is out of bounds.
    ///
    /// ```
    /// use cap_vec::CapVec;
    ///
    /// let mut v = CapVec::<i32, 3>::new();
    /// v.extend([10, 20, 30]);
    /// assert_eq!(v.remove(1), Some(20));
    /// assert_eq!(v.as_slice(), &[10, 30]);
    /// ```
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }

        debug_assert!(self.buf.is_some());
        let buf = unsafe { self.buf.as_mut().unwrap_unchecked() };
        let ptr = buf.as_mut_ptr();

        // Read the element to be removed
        let element = unsafe { buf[index].assume_init_read() };

        // Decrement the len of the array
        self.len -= 1;

        // Shift elements to fill the gap
        unsafe {
            core::ptr::copy(ptr.add(index + 1), ptr.add(index), self.len - index);
        }

        Some(element)
    }

    /// Returns a shared slice over the initialized portion of the buffer.
    pub fn as_slice(&self) -> &[T] {
        self.deref()
    }

    /// Returns a mutable slice over the initialized portion of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.deref_mut()
    }

    /// Returns the compile-time capacity of the vector.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns the current number of initialized elements.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Removes the subslice indicated by the given range from the vector,
    /// returning a double-ended iterator over the removed subslice.
    ///
    /// If the iterator is dropped before being fully consumed,
    /// it drops the remaining removed elements.
    ///
    /// The returned iterator keeps a mutable borrow on the vector to optimize
    /// its implementation.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    /// # Leaking
    ///
    /// If the returned iterator goes out of scope without being dropped (due to
    /// [`mem::forget`], for example), the vector may have lost and leaked
    /// elements arbitrarily, including elements outside the range.
    ///
    /// # Examples
    ///
    /// ```
    /// use cap_vec::CapVec;
    ///
    /// let mut v = CapVec::from([1, 2, 3]);
    /// let u: Vec<_> = v.drain(1..v.len()).collect();
    /// assert_eq!(v.as_slice(), &[1]);
    /// assert_eq!(u, &[2, 3]);
    ///
    /// // A full range clears the vector, like `clear()` does
    /// v.drain(0..v.len());
    /// assert_eq!(v.as_slice(), &[]);
    /// ```
    pub fn drain(&mut self, range: Range<usize>) -> Drain<'_, T, N> {
        assert!(range.start <= range.end);
        assert!(range.end <= self.len);

        // Memory safety
        //
        // When the Drain created, it shortens the length of the source vector
        // to make sure no uninitialized or moved-from elements are accessible
        // at all if the Drain's destructor never gets to run.
        //
        // Drain will ptr::read out the values to remove.
        //
        // When finished, remaining tail of the vec is copied back to cover
        // the hole, and the vector length is restored to the new length.
        let original_len = self.len();
        self.len = range.start;

        Drain {
            leftover_range: range.clone(),
            drop_range: range,
            vec: self,
            original_len,
        }
    }

    /// Returns an iterator over immutable references to the elements.
    pub fn iter(&self) -> Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns an iterator over mutable references to the elements.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// Drops all elements and resets the vector to an empty state.
    /// The capacity remains allocated.
    ///
    /// ```
    /// use cap_vec::CapVec;
    ///
    /// let mut v = CapVec::<i32, 4>::new();
    /// v.extend([1, 2, 3]);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    pub fn clear(&mut self) {
        if let Some(slice) = self.buf.as_mut() {
            slice[..self.len]
                .iter_mut()
                .for_each(|e| unsafe { e.assume_init_drop() });
        }

        self.len = 0;
    }
}

impl<T, const N: usize> Deref for CapVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        if let Some(elements) = self.buf.as_ref() {
            return unsafe { core::slice::from_raw_parts(elements.as_ptr() as *const T, self.len) };
        }

        &[]
    }
}

impl<T, const N: usize> DerefMut for CapVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if let Some(elements) = self.buf.as_mut() {
            return unsafe {
                core::slice::from_raw_parts_mut(elements.as_mut_ptr() as *mut T, self.len)
            };
        }

        &mut []
    }
}

impl<T, const N: usize> IntoIterator for CapVec<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    fn into_iter(mut self) -> Self::IntoIter {
        match self.buf.take() {
            None => IntoIter::default(),
            Some(buf) => IntoIter {
                inner: buf.into_iter().take(self.len),
            },
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a CapVec<T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut CapVec<T, N> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, const N: usize> PartialEq for CapVec<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T, const N: usize> PartialEq<[T]> for CapVec<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice() == other
    }
}

impl<T, const N: usize> PartialEq<&[T]> for CapVec<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, const N: usize> PartialEq<[T; N]> for CapVec<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &[T; N]) -> bool {
        self.as_slice() == other
    }
}

impl<T, const N: usize> PartialEq<&[T; N]> for CapVec<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &&[T; N]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, const N: usize> Eq for CapVec<T, N> where T: Eq {}

impl<T, const N: usize> PartialOrd for CapVec<T, N>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other)
    }
}

impl<T, const N: usize> PartialOrd<[T]> for CapVec<T, N>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &[T]) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other)
    }
}

impl<T, const N: usize> PartialOrd<[T; N]> for CapVec<T, N>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &[T; N]) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other)
    }
}

impl<T, const N: usize> Ord for CapVec<T, N>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_slice().cmp(other)
    }
}

impl<T, const N: usize> Hash for CapVec<T, N>
where
    T: Hash,
{
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T, const N: usize> core::fmt::Debug for CapVec<T, N>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T, const N: usize> Drop for CapVec<T, N> {
    fn drop(&mut self) {
        self.clear();
    }
}

// ---

pub struct Drain<'a, T, const N: usize> {
    leftover_range: Range<usize>,
    drop_range: Range<usize>,
    vec: &'a mut CapVec<T, N>,
    original_len: usize,
}

impl<'a, T, const N: usize> Iterator for Drain<'a, T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.leftover_range.next()?;
        debug_assert!(self.vec.buf.is_some());
        let element = unsafe { self.vec.buf.as_mut().unwrap_unchecked()[index].assume_init_read() };
        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.leftover_range.size_hint()
    }
}

impl<T, const N: usize> DoubleEndedIterator for Drain<'_, T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = self.leftover_range.next_back()?;
        debug_assert!(self.vec.buf.is_some());
        let element = unsafe { self.vec.buf.as_mut().unwrap_unchecked()[index].assume_init_read() };
        Some(element)
    }
}

impl<T, const N: usize> FusedIterator for Drain<'_, T, N> {}

impl<T, const N: usize> ExactSizeIterator for Drain<'_, T, N> {
    fn len(&self) -> usize {
        self.leftover_range.len()
    }
}

impl<T, const N: usize> Drop for Drain<'_, T, N> {
    fn drop(&mut self) {
        if let Some(buf) = self.vec.buf.as_mut() {
            for index in self.leftover_range.clone() {
                unsafe { buf[index].assume_init_drop() };
            }

            let ptr = buf.as_mut_ptr();

            unsafe {
                core::ptr::copy(
                    ptr.add(self.drop_range.end),
                    ptr.add(self.drop_range.start),
                    self.original_len - self.drop_range.end,
                );
            }
        }

        self.vec.len = self.original_len - self.drop_range.len();
    }
}

// ---

pub struct IntoIter<T, const N: usize> {
    inner: core::iter::Take<core::array::IntoIter<MaybeUninit<T>, N>>,
}

impl<T, const N: usize> Default for IntoIter<T, N> {
    fn default() -> Self {
        Self {
            inner: [const { MaybeUninit::uninit() }; N].into_iter().take(0),
        }
    }
}

impl<T, const N: usize> core::fmt::Debug for IntoIter<T, N>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IntoIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|e| unsafe { e.assume_init() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|e| unsafe { e.assume_init() })
    }
}

impl<T, const N: usize> FusedIterator for IntoIter<T, N> {}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        for mut element in self.inner.by_ref() {
            unsafe { element.assume_init_drop() };
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    use super::CapVec;

    #[test]
    fn test_default() {
        let sut = CapVec::<i32, 8>::default();
        assert_eq!(sut.len(), 0, "CapVec should start empty");
        assert_eq!(
            sut.capacity(),
            8,
            "Capacity must match compile-time constant"
        );
        assert!(sut.is_empty(), "CapVec should be empty by default");
        assert!(sut.as_slice().is_empty(), "Slice view should also be empty");
    }

    #[test]
    fn test_from_array() {
        const SEED: [i32; 4] = [1, 2, 3, 4];

        let sut = CapVec::<i32, 4>::from(SEED);
        assert_eq!(sut.len(), 4, "Length should equal array length");
        assert_eq!(sut.capacity(), 4, "Capacity should equal compile-time N");
        assert!(!sut.is_empty(), "CapVec from array should not be empty");
        assert_eq!(sut.as_slice(), &SEED, "Contents must match source array");
    }

    #[test]
    fn test_clone_non_empty_vec() {
        let mut base = CapVec::<i32, 4>::new();
        base.extend([1, 2, 3]);

        let sut = base.clone();
        assert_eq!(sut.len(), base.len(), "Clone should preserve length");
        assert_eq!(
            sut.capacity(),
            base.capacity(),
            "Clone should preserve capacity"
        );
        assert_eq!(sut.as_slice(), base.as_slice(), "Clone contents must match");

        base.push(4).unwrap();
        assert_ne!(
            sut.len(),
            base.len(),
            "Clone must not reflect later mutations"
        );
        assert_eq!(
            sut.as_slice(),
            &[1, 2, 3],
            "Clone contents must remain unaffected after modifying original"
        );
    }

    #[test]
    fn test_clone_empty_vec() {
        let mut base = CapVec::<i32, 5>::new();
        let sut = base.clone();
        assert!(sut.is_empty(), "Clone of empty CapVec should be empty");
        assert_eq!(sut.capacity(), 5);
        assert!(sut.as_slice().is_empty());

        base.push(0).unwrap();
        assert_ne!(
            sut.len(),
            base.len(),
            "Clone must not reflect later mutations"
        );
        assert_eq!(
            sut.as_slice(),
            &[],
            "Clone contents must remain unaffected after modifying original"
        );
    }

    #[test]
    fn test_new() {
        let sut = CapVec::<i32, 8>::new();
        assert_eq!(sut.len(), 0, "CapVec should start empty");
        assert_eq!(
            sut.capacity(),
            8,
            "Capacity must match compile-time constant"
        );
        assert!(sut.is_empty(), "CapVec should be empty by default");
        assert!(sut.as_slice().is_empty(), "Slice view should also be empty");
    }

    #[test]
    fn test_len() {
        let mut sut = CapVec::<i32, 3>::new();
        assert_eq!(sut.len(), 0, "CapVec should start empty");
        assert_eq!(
            sut.capacity(),
            3,
            "Capacity must match compile-time constant"
        );
        assert!(sut.is_empty(), "CapVec should be empty by default");

        sut.push(10).unwrap();
        assert_eq!(sut.len(), 1, "Length should increase after one push");

        sut.push(20).unwrap();
        assert_eq!(
            sut.len(),
            2,
            "Length should update correctly after two pushes"
        );

        sut.pop();
        assert_eq!(sut.len(), 1, "Length should decrease after a pop");

        sut.pop();
        assert_eq!(sut.len(), 0, "Length should be zero after the last pop");
    }

    #[test]
    fn test_capacity() {
        let sut = CapVec::<(), 4>::new();
        assert_eq!(
            sut.capacity(),
            4,
            "Capacity must match compile-time constant"
        );

        let sut = CapVec::<char, 16>::new();
        assert_eq!(
            sut.capacity(),
            16,
            "Capacity must match compile-time constant"
        );

        let sut = CapVec::<i64, 1>::new();
        assert_eq!(
            sut.capacity(),
            1,
            "Capacity must match compile-time constant"
        );
    }

    #[test]
    fn test_is_empty() {
        let mut sut = CapVec::<i32, 2>::new();
        assert!(sut.is_empty(), "CapVec should be empty by default");

        sut.push(1).unwrap();
        assert!(!sut.is_empty(), "CapVec should not be empty after push");

        sut.clear();
        assert!(sut.is_empty(), "CapVec should be empty again after clear()");
    }

    #[test]
    fn test_clear() {
        let mut sut = CapVec::<Box<str>, 3>::new();
        sut.push("a".into()).unwrap();
        sut.push("b".into()).unwrap();
        assert_eq!(sut.len(), 2, "Should have two elements before clear");

        sut.clear();
        assert_eq!(sut.len(), 0, "Clear should reset length to zero");
        assert!(sut.is_empty(), "CapVec should be empty after clear");

        // Ensure subsequent reuse works fine
        sut.push("c".into()).unwrap();
        assert_eq!(
            sut.as_slice(),
            &["c".into()],
            "CapVec should accept new elements after clear()"
        );
    }

    #[test]
    fn test_as_slice() {
        let mut sut = CapVec::<i32, 3>::new();
        sut.push(10).unwrap();
        sut.push(20).unwrap();
        sut.push(30).unwrap();

        let slice = sut.as_slice();
        assert_eq!(slice, &[10, 20, 30], "Slice should match inserted elements");
    }

    #[test]
    fn test_as_mut_slice() {
        let mut sut = CapVec::<i32, 4>::new();
        sut.extend([1, 2, 3]);

        {
            let slice = sut.as_mut_slice();
            slice[0] = 10;
            slice[1] = 20;
            slice[2] = 30;
        }

        assert_eq!(
            sut.as_slice(),
            &[10, 20, 30],
            "Mutable slice should allow modifying elements"
        );

        // Mutate using standard slice APIs
        sut.as_mut_slice().reverse();
        assert_eq!(
            sut.as_slice(),
            &[30, 20, 10],
            "Reversal via as_mut_slice() should apply"
        );
    }

    // ------------------------------------------------------------------------

    #[test]
    fn test_partially_fill_empty_vec_zst() {
        let mut sut = CapVec::<(), 4>::new();
        assert!(sut.is_empty());
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.as_slice(), []);

        let mut leftover = sut.extend(core::iter::repeat(()).take(2));
        assert_eq!(leftover.next(), None);

        assert!(!sut.is_empty());
        assert_eq!(sut.len(), 2);
        assert_eq!(sut.as_slice(), [(); 2]);
    }

    #[test]
    fn test_fill_empty_vec_zst() {
        let mut sut = CapVec::<(), 4>::new();
        assert!(sut.is_empty());
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.as_slice(), []);

        let mut leftover = sut.extend(core::iter::repeat(()));
        assert_eq!(leftover.next(), Some(()));

        assert!(!sut.is_empty());
        assert_eq!(sut.len(), 4);
        assert_eq!(sut.as_slice(), [(); 4]);
    }

    #[test]
    fn test_extend_partially_filled_vec_zst() {
        let mut sut = CapVec::<(), 8>::new();
        assert!(sut.is_empty());
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.as_slice(), []);

        let mut leftover = sut.extend(core::iter::repeat(()).take(3));
        assert_eq!(leftover.next(), None);

        assert!(!sut.is_empty());
        assert_eq!(sut.len(), 3);
        assert_eq!(sut.as_slice(), [(); 3]);

        let mut leftover = sut.extend(core::iter::repeat(()).take(3));
        assert_eq!(leftover.next(), None);

        assert!(!sut.is_empty());
        assert_eq!(sut.len(), 6);
        assert_eq!(sut.as_slice(), [(); 6]);

        let mut leftover = sut.extend(core::iter::repeat(()));
        assert_eq!(leftover.next(), Some(()));

        assert!(!sut.is_empty());
        assert_eq!(sut.len(), 8);
        assert_eq!(sut.as_slice(), [(); 8]);
    }

    #[test]
    fn test_extend_zero_capacity_vec() {
        let mut sut = CapVec::<(), 0>::new();
        assert!(sut.is_empty());
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.as_slice(), []);

        let mut leftover = sut.extend(core::iter::repeat(()));
        assert_eq!(leftover.next(), Some(()));
        assert!(sut.is_empty());
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.as_slice(), []);
    }

    // ------------------------------------------------------------------------

    #[test]
    fn test_insert() {
        let mut sut: CapVec<i64, 6> = CapVec::new();
        assert_eq!(sut.len(), 0);
        assert!(sut.is_empty());

        assert_eq!(Ok(()), sut.insert(0, 10));
        assert_eq!(sut.get(0), Some(&10));
        assert_eq!(sut.len(), 1);

        assert_eq!(Ok(()), sut.insert(1, 15));
        assert_eq!(sut.get(0), Some(&10));
        assert_eq!(sut.get(1), Some(&15));
        assert_eq!(sut.len(), 2);

        assert_eq!(Ok(()), sut.insert(0, 5));
        assert_eq!(sut.get(0), Some(&5));
        assert_eq!(sut.get(1), Some(&10));
        assert_eq!(sut.get(2), Some(&15));
        assert_eq!(sut.len(), 3);

        assert_eq!(Ok(()), sut.insert(3, 20));
        assert_eq!(sut.get(0), Some(&5));
        assert_eq!(sut.get(1), Some(&10));
        assert_eq!(sut.get(2), Some(&15));
        assert_eq!(sut.get(3), Some(&20));
        assert_eq!(sut.len(), 4);

        assert_eq!(Ok(()), sut.insert(2, 13));
        assert_eq!(sut.get(0), Some(&5));
        assert_eq!(sut.get(1), Some(&10));
        assert_eq!(sut.get(2), Some(&13));
        assert_eq!(sut.get(3), Some(&15));
        assert_eq!(sut.get(4), Some(&20));
        assert_eq!(sut.len(), 5);

        assert_eq!(Ok(()), sut.insert(4, 17));
        assert_eq!(sut.get(0), Some(&5));
        assert_eq!(sut.get(1), Some(&10));
        assert_eq!(sut.get(2), Some(&13));
        assert_eq!(sut.get(3), Some(&15));
        assert_eq!(sut.get(4), Some(&17));
        assert_eq!(sut.get(5), Some(&20));
        assert_eq!(sut.len(), 6);

        assert_eq!(Err(100), sut.insert(6, 100));
        assert_eq!(Err(100), sut.insert(usize::MAX, 100));
    }

    #[test]
    fn test_push() {
        let mut sut: CapVec<i64, 4> = CapVec::new();
        assert_eq!(sut.len(), 0);
        assert!(sut.is_empty());
        assert_eq!(sut, &[][..]);

        sut.push(0).unwrap();
        assert_eq!(sut.len(), 1);
        assert!(!sut.is_empty());
        assert_eq!(sut, &[0][..]);

        sut.push(1).unwrap();
        assert_eq!(sut.len(), 2);
        assert!(!sut.is_empty());
        assert_eq!(sut, &[0, 1][..]);

        sut.push(2).unwrap();
        assert_eq!(sut.len(), 3);
        assert!(!sut.is_empty());
        assert_eq!(sut, &[0, 1, 2][..]);

        sut.push(3).unwrap();
        assert_eq!(sut.len(), 4);
        assert!(!sut.is_empty());
        assert_eq!(sut, &[0, 1, 2, 3][..]);

        assert_eq!(sut.push(4), Err(4));
        assert_eq!(sut.len(), 4);
        assert!(!sut.is_empty());
        assert_eq!(sut, &[0, 1, 2, 3][..]);
    }

    #[test]
    fn test_remove() {
        let mut sut: CapVec<i64, 6> = CapVec::new();
        assert_eq!(sut.len(), 0);
        assert!(sut.is_empty());

        sut.insert(0, 0).unwrap();
        sut.insert(1, 1).unwrap();
        sut.insert(2, 2).unwrap();
        sut.insert(3, 3).unwrap();
        sut.insert(4, 4).unwrap();
        sut.insert(5, 5).unwrap();
        assert_eq!(sut.len(), 6);

        assert_eq!(sut.remove(2), Some(2));
        assert_eq!(sut.len(), 5);
        assert_eq!(sut.get(0), Some(&0));
        assert_eq!(sut.get(1), Some(&1));
        assert_eq!(sut.get(2), Some(&3));
        assert_eq!(sut.get(3), Some(&4));
        assert_eq!(sut.get(4), Some(&5));

        assert_eq!(sut.remove(3), Some(4));
        assert_eq!(sut.len(), 4);
        assert_eq!(sut.get(0), Some(&0));
        assert_eq!(sut.get(1), Some(&1));
        assert_eq!(sut.get(2), Some(&3));
        assert_eq!(sut.get(3), Some(&5));

        assert_eq!(sut.remove(1), Some(1));
        assert_eq!(sut.len(), 3);
        assert_eq!(sut.get(0), Some(&0));
        assert_eq!(sut.get(1), Some(&3));
        assert_eq!(sut.get(2), Some(&5));

        assert_eq!(sut.remove(0), Some(0));
        assert_eq!(sut.len(), 2);
        assert_eq!(sut.get(0), Some(&3));
        assert_eq!(sut.get(1), Some(&5));

        assert_eq!(sut.remove(1), Some(5));
        assert_eq!(sut.len(), 1);
        assert_eq!(sut.get(0), Some(&3));

        assert_eq!(sut.remove(0), Some(3));
        assert_eq!(sut.len(), 0);

        assert_eq!(sut.remove(0), None);
    }

    #[test]
    fn test_pop() {
        let mut sut: CapVec<i64, 4> = CapVec::new();

        let mut leftover = sut.extend(0..5);
        assert_eq!(leftover.next(), Some(4));
        assert_eq!(sut.len(), 4);
        assert!(!sut.is_empty());

        for i in (0..4).rev() {
            assert_eq!(sut.pop(), Some(i));
            assert_eq!(sut.len(), i as usize);
        }

        assert_eq!(sut.pop(), None);
        assert_eq!(sut.len(), 0);
        assert!(sut.is_empty());
    }

    #[test]
    fn test_mutability() {
        let mut sut: CapVec<i64, 4> = CapVec::new();

        let mut leftover = sut.extend(0..5);
        assert_eq!(leftover.next(), Some(4));

        assert_eq!(sut.len(), 4);
        assert!(!sut.is_empty());
        assert_eq!(sut, [0, 1, 2, 3]);

        sut[0] *= 10;
        sut[1] *= 10;
        sut[2] *= 10;
        sut[3] *= 10;

        assert_eq!(sut, [0, 10, 20, 30]);
        assert_eq!(sut.len(), 4);
        assert!(!sut.is_empty());
    }

    #[test]
    fn test_get_on_empty_vec() {
        let sut: CapVec<i64, 6> = CapVec::new();
        assert_eq!(sut.len(), 0);
        assert!(sut.is_empty());
        assert_eq!(sut.get(0), None);
    }

    #[test]
    fn test_get_mut_on_empty_vec() {
        let mut sut: CapVec<i64, 6> = CapVec::new();
        assert_eq!(sut.len(), 0);
        assert!(sut.is_empty());
        assert_eq!(sut.get_mut(0), None);
    }

    #[test]
    fn test_comparison() {
        let sut: CapVec<i64, 4> = CapVec::from([0, 1, 2, 3]);

        assert_eq!(sut, sut);
        assert_eq!(sut, sut.clone());

        assert_eq!(sut, [0, 1, 2, 3]);
        assert_eq!(sut, &[0, 1, 2, 3]);
        assert_eq!(sut, [0, 1, 2, 3][..]);
        assert_eq!(sut, &[0, 1, 2, 3][..]);
        assert_eq!(sut, CapVec::from([0, 1, 2, 3]));

        assert_ne!(sut, [0, 10, 20, 30]);
        assert_ne!(sut, &[0, 10, 20, 30]);
        assert_ne!(sut, [0, 10, 20, 30][..]);
        assert_ne!(sut, &[0, 10, 20, 30][..]);
        assert_ne!(sut, CapVec::from([0, 10, 20, 30]));

        assert_ne!(sut, [][..]);
        assert_ne!(sut, &[][..]);

        let sut = CapVec::<i32, 4>::new();
        assert_eq!(sut, sut);
        assert_eq!(sut, sut.clone());

        assert_eq!(sut, [][..]);
        assert_eq!(sut, &[][..]);
        assert_eq!(sut, CapVec::new());
    }

    #[test]
    fn test_drain_partially() {
        let mut sut = CapVec::<_, 8>::new();
        let mut leftover = sut.extend((0..8).map(Box::new));
        assert_eq!(leftover.next(), None);

        assert_eq!(sut.len(), 8);
        assert!(!sut.is_empty());
        assert_eq!(sut, [0, 1, 2, 3, 4, 5, 6, 7].map(Box::new));

        let iter = sut.drain(1..7);
        assert_eq!(iter.take(2).collect::<Vec<_>>(), [1, 2].map(Box::new));

        assert_eq!(sut.len(), 2);
        assert!(!sut.is_empty());
        assert_eq!(sut.as_slice(), [0, 7].map(Box::new));
    }

    #[test]
    fn test_drain_from_both_ends() {
        let mut sut = CapVec::<_, 8>::new();
        let mut leftover = sut.extend((0..8).map(Box::new));
        assert_eq!(leftover.next(), None);

        assert_eq!(sut.len(), 8);
        assert!(!sut.is_empty());
        assert_eq!(sut, [0, 1, 2, 3, 4, 5, 6, 7].map(Box::new));

        let mut iter = sut.drain(1..7);
        assert_eq!(iter.next(), Some(Box::new(1)));
        assert_eq!(iter.next_back(), Some(Box::new(6)));

        assert_eq!(iter.next_back(), Some(Box::new(5)));
        assert_eq!(iter.next(), Some(Box::new(2)));

        assert_eq!(iter.len(), 2);
    }

    #[test]
    fn test_drain_all() {
        let mut sut = CapVec::<_, 8>::new();
        let mut leftover = sut.extend((0..8).map(Box::new));
        assert_eq!(leftover.next(), None);

        assert_eq!(sut.len(), 8);
        assert!(!sut.is_empty());
        assert_eq!(sut, [0, 1, 2, 3, 4, 5, 6, 7].map(Box::new));

        let iter = sut.drain(0..8);
        assert_eq!(
            iter.collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5, 6, 7].map(Box::new)
        );

        assert_eq!(sut.len(), 0);
        assert!(sut.is_empty());
        assert_eq!(sut.as_slice(), []);
    }

    #[test]
    fn test_drain_partially_zst() {
        let mut sut = CapVec::<_, 8>::new();
        let mut leftover = sut.extend([(); 8]);
        assert_eq!(leftover.next(), None);

        assert_eq!(sut.len(), 8);
        assert!(!sut.is_empty());
        assert_eq!(sut, [(); 8]);

        let iter = sut.drain(1..7);
        assert_eq!(iter.take(2).collect::<Vec<_>>(), [(); 2]);

        assert_eq!(sut.len(), 2);
        assert!(!sut.is_empty());
        assert_eq!(sut.as_slice(), [(); 2]);
    }

    #[test]
    fn test_drain_all_zst() {
        let mut sut = CapVec::<_, 8>::new();
        let mut leftover = sut.extend([(); 8]);
        assert_eq!(leftover.next(), None);

        assert_eq!(sut.len(), 8);
        assert!(!sut.is_empty());
        assert_eq!(sut, [(); 8]);

        let iter = sut.drain(0..8);
        assert_eq!(iter.collect::<Vec<_>>(), [(); 8]);

        assert_eq!(sut.len(), 0);
        assert!(sut.is_empty());
        assert_eq!(sut.as_slice(), []);
    }

    #[test]
    fn test_iter_empty() {
        let base = CapVec::<i32, 4>::new();
        let mut sut = base.iter();
        assert_eq!(sut.len(), 0, "Iterator over empty CapVec should have len 0");
        assert_eq!(
            sut.next(),
            None,
            "Iterator over empty CapVec should return None"
        );
        assert_eq!(
            sut.clone().count(),
            0,
            "Cloned iterator should also be empty"
        );
    }

    #[test]
    fn test_iter_basic_forward_iteration() {
        let mut base = CapVec::<i32, 5>::new();
        base.extend([10, 20, 30]);

        let mut sut = base.iter();
        assert_eq!(sut.next(), Some(&10));
        assert_eq!(sut.next(), Some(&20));
        assert_eq!(sut.next(), Some(&30));
        assert_eq!(sut.next(), None);
    }

    #[test]
    fn test_iter_clone() {
        let base = CapVec::<i32, 3>::from([1, 2, 3]);
        let mut sut1 = base.iter();
        let mut sut2 = sut1.clone();
        assert_eq!(sut1.next(), Some(&1));
        assert_eq!(sut2.next(), Some(&1));
        assert_eq!(sut1.next(), Some(&2));
        assert_eq!(sut2.next(), Some(&2));
        assert_eq!(sut1.next(), Some(&3));
        assert_eq!(sut2.next(), Some(&3));
        assert_eq!(sut1.next(), None);
        assert_eq!(sut2.next(), None);
    }

    #[test]
    fn test_iter_collect() {
        let base = CapVec::<u8, 4>::from([5, 6, 7, 8]);
        let collected: Vec<_> = base.iter().copied().collect();
        assert_eq!(
            collected,
            [5, 6, 7, 8],
            "Iter should produce same items as Vec"
        );

        assert!(
            base.iter().eq([5, 6, 7, 8].iter()),
            "CapVec::iter must produce same sequence as Vec::iter"
        );
    }

    #[test]
    fn test_iter_double_ended_iteration() {
        let base = CapVec::<i32, 4>::from([1, 2, 3, 4]);
        let mut sut = base.iter();

        assert_eq!(sut.next_back(), Some(&4));
        assert_eq!(sut.next(), Some(&1));
        assert_eq!(sut.next_back(), Some(&3));
        assert_eq!(sut.next(), Some(&2));
        assert_eq!(sut.next_back(), None);
        assert_eq!(sut.next(), None);
    }

    #[test]
    fn test_iter_fused_behaviour() {
        let base = CapVec::<i32, 3>::from([9, 8, 7]);
        let mut sut = base.iter();
        for _ in 0..3 {
            sut.next();
        }
        assert_eq!(
            sut.next(),
            None,
            "Iterator should yield None after exhaustion"
        );
        assert_eq!(
            sut.next(),
            None,
            "Iterator should stay fused (repeated None)"
        );
    }

    #[test]
    fn test_iter_len_consistency() {
        let base = CapVec::<i32, 4>::from([10, 20, 30, 40]);
        let mut sut = base.iter();
        assert_eq!(sut.len(), 4);
        assert_eq!(sut.next(), Some(&10));
        assert_eq!(sut.len(), 3);
        assert_eq!(sut.next_back(), Some(&40));
        assert_eq!(sut.len(), 2);
        assert_eq!(sut.next(), Some(&20));
        assert_eq!(sut.len(), 1);
        assert_eq!(sut.next_back(), Some(&30));
        assert_eq!(sut.len(), 0);
    }

    #[test]
    fn test_iter_mut_empty() {
        let mut base = CapVec::<i32, 4>::new();
        let mut sut = base.iter_mut();
        assert_eq!(sut.len(), 0, "Iterator over empty CapVec should have len 0");
        assert_eq!(
            sut.next(),
            None,
            "Iterator over empty CapVec should yield None"
        );
        assert_eq!(
            sut.next_back(),
            None,
            "Double-ended iterator should yield None on empty CapVec"
        );
    }

    #[test]
    fn test_iter_mut_forward_iteration_and_mutation() {
        let mut base = CapVec::<i32, 5>::new();
        base.extend([1, 2, 3]);

        for x in base.iter_mut() {
            *x *= 10;
        }
        assert_eq!(
            base.as_slice(),
            &[10, 20, 30],
            "Mutable iterator should allow modifying items in order"
        );
    }

    #[test]
    fn test_iter_mut_double_ended_iteration() {
        let mut base = CapVec::<i32, 4>::from([1, 2, 3, 4]);
        let mut sut = base.iter_mut();
        assert_eq!(sut.next_back().map(|x| *x), Some(4));
        assert_eq!(sut.next().map(|x| *x), Some(1));
        assert_eq!(sut.next_back().map(|x| *x), Some(3));
        assert_eq!(sut.next().map(|x| *x), Some(2));
        assert_eq!(sut.next_back(), None);
        assert_eq!(sut.next(), None);
    }

    #[test]
    fn test_iter_mut_backward_iteration_and_mutation() {
        let mut base = CapVec::<i32, 5>::new();
        base.extend([1, 2, 3]);

        for x in base.iter_mut().rev() {
            *x *= 10;
        }
        assert_eq!(
            base.as_slice(),
            &[10, 20, 30],
            "Mutable iterator should allow modifying items in order"
        );
    }

    #[test]
    fn test_iter_mut_double_ended_iteration_and_mutation() {
        let mut base = CapVec::<i32, 4>::from([1, 2, 3, 4]);
        let mut sut = base.iter_mut();

        if let Some(front) = sut.next() {
            *front *= 2;
        }

        if let Some(back) = sut.next_back() {
            *back *= 3;
        }

        assert_eq!(
            base.as_slice(),
            &[2, 2, 3, 12],
            "Mutations through iter_mut should persist"
        );
    }

    #[test]
    fn test_iter_mut_vs_vec_iter_mut() {
        let mut base = CapVec::<u8, 4>::from([5, 6, 7, 8]);
        let mut v = [5, 6, 7, 8].to_vec();

        for (b, w) in base.iter_mut().zip(v.iter_mut()) {
            *b += 1;
            *w += 1;
        }

        assert_eq!(
            base.as_slice(),
            v.as_slice(),
            "CapVec::iter_mut should behave like Vec::iter_mut"
        );
    }

    #[test]
    fn test_iter_mut_len_consistency() {
        let mut base = CapVec::<i32, 4>::from([10, 20, 30, 40]);
        let mut sut = base.iter_mut();
        assert_eq!(sut.len(), 4);
        assert_eq!(sut.next(), Some(&mut 10));
        assert_eq!(sut.len(), 3);
        assert_eq!(sut.next_back(), Some(&mut 40));
        assert_eq!(sut.len(), 2);
        assert_eq!(sut.next(), Some(&mut 20));
        assert_eq!(sut.len(), 1);
        assert_eq!(sut.next_back(), Some(&mut 30));
        assert_eq!(sut.len(), 0);
    }

    #[test]
    fn test_iter_mut_fused_behavior() {
        let mut base = CapVec::<i32, 3>::from([9, 8, 7]);
        let mut sut = base.iter_mut();

        for _ in 0..3 {
            sut.next();
        }

        assert_eq!(
            sut.next(),
            None,
            "Iterator should yield None after exhaustion"
        );

        assert_eq!(
            sut.next(),
            None,
            "Iterator should stay fused (repeated None)"
        );
    }

    #[test]
    fn test_into_iter_empty() {
        let base = CapVec::<i32, 4>::new();
        let mut sut = base.into_iter();

        assert_eq!(sut.len(), 0, "Iterator from empty CapVec should have len 0");
        assert_eq!(
            sut.next(),
            None,
            "Iterator from empty CapVec should yield None"
        );
        assert_eq!(
            sut.next_back(),
            None,
            "Double-ended iterator should yield None on empty CapVec"
        );
    }

    #[test]
    fn test_into_iter_basic_forward_iteration() {
        let mut base = CapVec::<i32, 5>::new();
        base.extend([1, 2, 3]);

        let mut sut = base.into_iter();
        assert_eq!(sut.next(), Some(1));
        assert_eq!(sut.next(), Some(2));
        assert_eq!(sut.next(), Some(3));
        assert_eq!(sut.next(), None);
    }

    #[test]
    fn test_into_iter_double_ended_iteration() {
        let base = CapVec::<i32, 4>::from([10, 20, 30, 40]);

        let mut sut = base.into_iter();
        assert_eq!(sut.next_back(), Some(40));
        assert_eq!(sut.next(), Some(10));
        assert_eq!(sut.next_back(), Some(30));
        assert_eq!(sut.next(), Some(20));
        assert_eq!(sut.next_back(), None);
        assert_eq!(sut.next(), None);
    }

    #[test]
    fn test_into_iter_collect() {
        let base = CapVec::<u8, 4>::from([5, 6, 7, 8]);
        let collected: Vec<_> = base.into_iter().collect();
        assert_eq!(
            collected,
            [5, 6, 7, 8],
            "into_iter() should produce same items as Vec"
        );
    }

    #[test]
    fn test_into_iter_vs_vec_into_iter() {
        let base = CapVec::<i32, 4>::from([1, 2, 3, 4]);
        let v = [1, 2, 3, 4].to_vec();

        assert_eq!(
            base.into_iter().collect::<Vec<_>>(),
            v,
            "CapVec::into_iter() should yield identical results to Vec::into_iter()"
        );
    }

    #[test]
    fn test_into_iter_len_consistency() {
        let base = CapVec::<i32, 4>::from([10, 20, 30, 40]);
        let mut sut = base.into_iter();
        assert_eq!(sut.len(), 4);
        assert_eq!(sut.next(), Some(10));
        assert_eq!(sut.len(), 3);
        assert_eq!(sut.next_back(), Some(40));
        assert_eq!(sut.len(), 2);
        assert_eq!(sut.next(), Some(20));
        assert_eq!(sut.len(), 1);
        assert_eq!(sut.next_back(), Some(30));
        assert_eq!(sut.len(), 0);
    }

    #[test]
    fn test_into_iter_fused_behavior() {
        let base = CapVec::<i32, 3>::from([1, 2, 3]);
        let mut sut = base.into_iter();

        for _ in 0..3 {
            sut.next();
        }

        assert_eq!(
            sut.next(),
            None,
            "Iterator should yield None after exhaustion"
        );

        assert_eq!(
            sut.next(),
            None,
            "Iterator should remain fused (repeated None)"
        );
    }

    #[test]
    fn test_into_iter_double_ended_len_updates() {
        let base = CapVec::<i32, 5>::from([1, 2, 3, 4, 5]);
        let mut sut = base.into_iter();

        assert_eq!(sut.len(), 5);
        assert_eq!(sut.next(), Some(1));
        assert_eq!(sut.len(), 4);
        assert_eq!(sut.next_back(), Some(5));
        assert_eq!(sut.len(), 3);
        assert_eq!(sut.next(), Some(2));
        assert_eq!(sut.len(), 2);
        assert_eq!(sut.next_back(), Some(4));
        assert_eq!(sut.len(), 1);
        assert_eq!(sut.next(), Some(3));
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.next(), None);
    }

    #[test]
    fn test_into_iter_drop() {
        let base = CapVec::<Box<str>, 4>::from(["a", "b", "c", "d"].map(Box::from));

        let mut sut = base.into_iter();
        assert_eq!(sut.next().as_deref(), Some("a"));
        assert_eq!(sut.next_back().as_deref(), Some("d"));

        // ensure memory gets release correctly
        core::mem::drop(sut);
    }

    #[test]
    fn test_ref_into_iter_equals_iter() {
        let base = CapVec::from([0, 1, 2, 3]);
        assert!(base.iter().eq(&base));

        let base = CapVec::from([0; 0]);
        assert!(base.iter().eq(&base));

        let base = CapVec::from([(); 0]);
        assert!(base.iter().eq(&base));

        let base = CapVec::from([(); 5]);
        assert!(base.iter().eq(&base));
    }

    #[test]
    fn test_ref_mut_into_iter_equals_iter_mut() {
        let mut base = CapVec::from([0, 1, 2, 3]);
        assert!([0, 1, 2, 3].iter_mut().eq(&mut base));

        let mut base = CapVec::from([0; 0]);
        assert!([0; 0].iter_mut().eq(&mut base));

        let mut base = CapVec::from([(); 0]);
        assert!([(); 0].iter_mut().eq(&mut base));

        let mut base = CapVec::from([(); 5]);
        assert!([(); 5].iter_mut().eq(&mut base));
    }
}
