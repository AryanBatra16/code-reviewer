# Explanation Examples (Integrated Gradients)

## Example 1: CVE-2012-3525 (CWE-20)

**Top suspicious lines:**

```c
    [  6] (score=0.0571)          valid = 1;
    [  5] (score=0.0503)          log_write(in->s2s->log, LOG_NOTICE, "[%d] [%s, port=%d] incoming route '%s' is now valid%s%s", in->fd->fd, in->ip, in->port, rkey, (in->s->flags & SX_SSL_WRAPPER) ? ", TLS negotiated" : "", in->s->compressed ? ", ZLIB compression enabled" : "");
    [  4] (score=0.0250)          xhash_put(in->states, pstrdup(xhash_pool(in->states), rkey), (void *) conn_VALID);
    [  2] (score=0.0202)      attr = nad_find_attr(nad, 0, -1, "type", "valid");
    [  0] (score=0.0120)      rkey = s2s_route_key(NULL, to->domain, from->domain);
```

- Loc@1: ✗  Loc@3: ✗

---

## Example 2: CVE-2013-7339 (CWE-399)

**Top suspicious lines:**

```c
    [  1] (score=0.0193)  	/* due to this, we will claim to support iWARP devices unless we
    [  6] (score=0.0192)  	rdsdebug("addr %pI4 ret %d node type %d\n",
>>> [  3] (score=0.0184) 	if (ret || cm_id->device->node_type != RDMA_NODE_IB_CA)
    [  0] (score=0.0166)  	ret = rdma_bind_addr(cm_id, (struct sockaddr *)&sin);
    [  4] (score=0.0127)  		ret = -EADDRNOTAVAIL;
```

- Loc@1: ✗  Loc@3: ✓

---

## Example 3: CVE-2011-1759 (CWE-189)

**Top suspicious lines:**

```c
    [  5] (score=0.0402)  	sops = kmalloc(sizeof(*sops) * nsops, GFP_KERNEL);
>>> [  3] (score=0.0190) 	if (nsops < 1)
    [  4] (score=0.0133)  		return -EINVAL;
    [  6] (score=0.0128)  	if (!sops)
    [  1] (score=0.0051)  	int i;
```

- Loc@1: ✗  Loc@3: ✓

---

## Example 4: CVE-2011-1575 (CWE-399)

**Top suspicious lines:**

```c
    [  3] (score=0.0204)  int sfgets(void)
    [ 12] (score=0.0197)      if (scanned > (size_t) 0U) {       /* support pipelining */
>>> [ 10] (score=0.0085)     static size_t readnbd;
>>> [  9] (score=0.0085)     static size_t scanned;
    [  5] (score=0.0078)      struct pollfd pfd;
```

- Loc@1: ✗  Loc@3: ✓

---

## Example 5: CVE-2013-4127 (CWE-399)

**Top suspicious lines:**

```c
    [  2] (score=0.0395)  	if (oldubufs) {
    [  5] (score=0.0278)  		vhost_zerocopy_signal_used(n, vq);
    [  4] (score=0.0203)  		mutex_lock(&vq->mutex);
>>> [  3] (score=0.0171) 		vhost_net_ubuf_put_and_wait(oldubufs);
    [  6] (score=0.0169)  		mutex_unlock(&vq->mutex);
```

- Loc@1: ✗  Loc@3: ✗

---

