# Explanation Examples (Integrated Gradients)

## Example 1: CVE-2018-9498 (CWE-787)

**Top suspicious lines:**

```c
>>> [  5] (score=0.1035)     return sk_malloc_flags(size, SK_MALLOC_THROW);
>>> [ 17] (score=0.0878)     return (flags & SK_MALLOC_THROW) ? moz_xmalloc(size) : malloc(size);
    [  8] (score=0.0862)  void* sk_realloc_throw(void* addr, size_t size) {
>>> [  4] (score=0.0697) void* sk_malloc_throw(size_t size) {
    [  9] (score=0.0666)      return moz_xrealloc(addr, size);
```

- Loc@1: ✓  Loc@3: ✓

---

## Example 2: CVE-2015-1805 (CWE-17)

**Top suspicious lines:**

```c
>>> [  7] (score=0.2075) 			iov_fault_in_pages_read(iov, chars);
    [ 34] (score=0.0541)  			buf->ops = &anon_pipe_buf_ops;
>>> [ 14] (score=0.0407) 			error = pipe_iov_copy_from_user(src, iov, chars,
    [ 33] (score=0.0342)  			buf->page = page;
>>> [ 10] (score=0.0314) 				src = kmap_atomic(page);
```

- Loc@1: ✓  Loc@3: ✓

---

## Example 3: CVE-2017-7742 (CWE-119)

**Top suspicious lines:**

```c
    [  6] (score=0.0713)  } FLAC_PRIVATE ;
    [  0] (score=0.0631)  	unsigned bufferpos ;
    [  2] (score=0.0627)  	const FLAC__Frame *frame ;
>>> [  3] (score=0.0607) 	FLAC__bool bufferbackup ;
    [  5] (score=0.0395)  	unsigned compression ;
```

- Loc@1: ✗  Loc@3: ✗

---

## Example 4: CVE-2019-15226 (nan)

**Top suspicious lines:**

```c
    [ 13] (score=0.3147)  HeaderMapImpl::HeaderMapImpl() { memset(&inline_headers_, 0, sizeof(inline_headers_)); }
>>> [  3] (score=0.2021) void HeaderMapImpl::appendToHeader(HeaderString& header, absl::string_view data) {
    [  4] (score=0.0399)    if (data.empty()) {
    [  7] (score=0.0370)    if (!header.empty()) {
    [ 10] (score=0.0345)    header.append(data.data(), data.size());
```

- Loc@1: ✗  Loc@3: ✓

---

## Example 5: CVE-2016-1696 (CWE-284)

**Top suspicious lines:**

```c
    [  4] (score=0.0919)        base::Bind(&I18NCustomBindings::GetL10nMessage, base::Unretained(this)));
    [  7] (score=0.0620)                             base::Unretained(this)));
>>> [  8] (score=0.0564)   RouteFunction("DetectTextLanguage",
    [  6] (score=0.0491)                  base::Bind(&I18NCustomBindings::GetL10nUILanguage,
    [  9] (score=0.0490)                  base::Bind(&I18NCustomBindings::DetectTextLanguage,
```

- Loc@1: ✗  Loc@3: ✓

---

