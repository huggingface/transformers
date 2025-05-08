[![PyPI](https://img.shields.io/pypi/v/jwcrypto.svg)](https://pypi.org/project/jwcrypto/)
[![Changelog](https://img.shields.io/github/v/release/latchset/jwcrypto?label=changelog)](https://github.com/latchset/jwcrypto/releases)
[![Build Status](https://github.com/latchset/jwcrypto/actions/workflows/build.yml/badge.svg)](https://github.com/latchset/jwcrypto/actions/workflows/build.yml)
[![ppc64le Build](https://github.com/latchset/jwcrypto/actions/workflows/ppc64le.yml/badge.svg)](https://github.com/latchset/jwcrypto/actions/workflows/ppc64le.yml)
[![Code Scan](https://github.com/latchset/jwcrypto/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/latchset/jwcrypto/actions/workflows/codeql-analysis.yml)
[![Documentation Status](https://readthedocs.org/projects/jwcrypto/badge/?version=latest)](https://jwcrypto.readthedocs.io/en/latest/?badge=latest)

JWCrypto
========

An implementation of the JOSE Working Group documents:
- RFC 7515 - JSON Web Signature (JWS)
- RFC 7516 - JSON Web Encryption (JWE)
- RFC 7517 - JSON Web Key (JWK)
- RFC 7518 - JSON Web Algorithms (JWA)
- RFC 7519 - JSON Web Token (JWT)
- RFC 7520 - Examples of Protecting Content Using JSON Object Signing and
  Encryption (JOSE)

Installation
============

    pip install jwcrypto

Documentation
=============

http://jwcrypto.readthedocs.org

Deprecation Notices
===================

2020.12.11: The RSA1_5 algorithm is now considered deprecated due to numerous
implementation issues that make it a very problematic tool to use safely.
The algorithm can still be used but requires explicitly allowing it on object
instantiation. If your application depends on it there are examples of how to
re-enable RSA1_5 usage in the tests files.

Note: if you enable support for `RSA1_5` and the attacker can send you chosen
ciphertext and is able to measure the processing times of your application,
then your application will be vulnerable to a Bleichenbacher RSA padding
oracle, allowing the so-called "Million messages attack". That attack allows
to decrypt intercepted messages (even if they were encrypted with RSA-OAEP) or
forge signatures (both RSA-PKCS#1 v1.5 and RSASSA-PSS).

Given JWT is generally used in tokens to sign authorization assertions or to
encrypt private key material, this is a particularly severe issue, and must
not be underestimated.
