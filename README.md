# cap_vec

[![Crates.io](https://img.shields.io/crates/v/cap_vec.svg)](https://crates.io/crates/cap_vec)
[![Docs.rs](https://img.shields.io/docsrs/cap_vec)](https://docs.rs/cap_vec)
[![License](https://img.shields.io/crates/l/cap_vec.svg)](https://github.com/daddinuz/cap_vec/blob/main/LICENSE)

A **heap-allocated**, **fixed-capacity**, **variable-size** array, `no_std` compatible.

`CapVec<T, N>` provides a middle ground between stack-allocated arrays `[T; N]` and dynamically growing vectors `Vec<T>`.
It allocates a **heap-backed buffer of fixed capacity `N`**, but allows the logical length to grow or shrink dynamically, up to that capacity.

---

## ✨ Features

- ✅ **Fixed capacity** — set at compile time.
- ✅ **Heap allocation** — no stack overflow even for large `N`.
- ✅ **no_std compatible** — uses only `alloc`.
- ✅ Supports most common collection operations:
  - `push`, `pop`, `insert`, `remove`
  - `clear`, `extend`, `drain`
  - `iter`, `iter_mut`, and `into_iter`
- ✅ Safe, ergonomic API — mirrors `Vec<T>` where possible.
- ✅ Zero-cost iteration (implements `Iterator`, `DoubleEndedIterator`, `FusedIterator`).
- ✅ No hidden allocations after initialization.

---

## 🚀 Example

```rust
use cap_vec::CapVec;

fn main() {
    // Create a CapVec with capacity for 4 elements
    let mut cv = CapVec::<i32, 4>::new();

    // Push values
    cv.push(10).unwrap();
    cv.push(20).unwrap();
    cv.push(30).unwrap();

    assert_eq!(cv.len(), 3);
    assert_eq!(cv.capacity(), 4);
    assert_eq!(cv.as_slice(), &[10, 20, 30]);

    // Insert in the middle
    cv.insert(1, 15).unwrap();
    assert_eq!(cv.as_slice(), &[10, 15, 20, 30]);

    // Remove one element
    let removed = cv.remove(2);
    assert_eq!(removed, Some(20));
    assert_eq!(cv.as_slice(), &[10, 15, 30]);

    // Iterate immutably
    for x in cv.iter() {
        println!("{x}");
    }

    // Iterate mutably
    for x in cv.iter_mut() {
        *x *= 2;
    }

    assert_eq!(cv.as_slice(), &[20, 30, 60]);

    // Consume into an iterator
    let v: Vec<_> = cv.into_iter().collect();
    assert_eq!(v, vec![20, 30, 60]);
}
```

---

## 🔨 Installation

Add `cap_vec` to your `Cargo.toml`:

```bash
cargo add cap_vec
```

or edit your Cargo.toml manually by adding:

```toml
[dependencies]
cap_vec = "0.2"
```

## Safety and Coverage

This crate contains a small portion of unsafe code.  
All tests run under [miri](https://github.com/rust-lang/miri) and the tests cover about 80% of the code.  
You can generate the coverage report using [tarpaulin](https://github.com/xd009642/tarpaulin).

## Contributions

Contributions are always welcome! Feel free to open an issue or submit a pull request.

## License

This crate is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
