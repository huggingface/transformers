'use strict';

var test = require('tape');
var inspect = require('object-inspect');
var SaferBuffer = require('safer-buffer').Buffer;
var forEach = require('for-each');
var v = require('es-value-fixtures');

var utils = require('../lib/utils');

test('merge()', function (t) {
    t.deepEqual(utils.merge(null, true), [null, true], 'merges true into null');

    t.deepEqual(utils.merge(null, [42]), [null, 42], 'merges null into an array');

    t.deepEqual(utils.merge({ a: 'b' }, { a: 'c' }), { a: ['b', 'c'] }, 'merges two objects with the same key');

    var oneMerged = utils.merge({ foo: 'bar' }, { foo: { first: '123' } });
    t.deepEqual(oneMerged, { foo: ['bar', { first: '123' }] }, 'merges a standalone and an object into an array');

    var twoMerged = utils.merge({ foo: ['bar', { first: '123' }] }, { foo: { second: '456' } });
    t.deepEqual(twoMerged, { foo: { 0: 'bar', 1: { first: '123' }, second: '456' } }, 'merges a standalone and two objects into an array');

    var sandwiched = utils.merge({ foo: ['bar', { first: '123', second: '456' }] }, { foo: 'baz' });
    t.deepEqual(sandwiched, { foo: ['bar', { first: '123', second: '456' }, 'baz'] }, 'merges an object sandwiched by two standalones into an array');

    var nestedArrays = utils.merge({ foo: ['baz'] }, { foo: ['bar', 'xyzzy'] });
    t.deepEqual(nestedArrays, { foo: ['baz', 'bar', 'xyzzy'] });

    var noOptionsNonObjectSource = utils.merge({ foo: 'baz' }, 'bar');
    t.deepEqual(noOptionsNonObjectSource, { foo: 'baz', bar: true });

    var func = function f() {};
    t.deepEqual(
        utils.merge(func, { foo: 'bar' }),
        [func, { foo: 'bar' }],
        'functions can not be merged into'
    );

    func.bar = 'baz';
    t.deepEqual(
        utils.merge({ foo: 'bar' }, func),
        { foo: 'bar', bar: 'baz' },
        'functions can be merge sources'
    );

    t.test(
        'avoids invoking array setters unnecessarily',
        { skip: typeof Object.defineProperty !== 'function' },
        function (st) {
            var setCount = 0;
            var getCount = 0;
            var observed = [];
            Object.defineProperty(observed, 0, {
                get: function () {
                    getCount += 1;
                    return { bar: 'baz' };
                },
                set: function () { setCount += 1; }
            });
            utils.merge(observed, [null]);
            st.equal(setCount, 0);
            st.equal(getCount, 1);
            observed[0] = observed[0]; // eslint-disable-line no-self-assign
            st.equal(setCount, 1);
            st.equal(getCount, 2);
            st.end();
        }
    );

    t.end();
});

test('assign()', function (t) {
    var target = { a: 1, b: 2 };
    var source = { b: 3, c: 4 };
    var result = utils.assign(target, source);

    t.equal(result, target, 'returns the target');
    t.deepEqual(target, { a: 1, b: 3, c: 4 }, 'target and source are merged');
    t.deepEqual(source, { b: 3, c: 4 }, 'source is untouched');

    t.end();
});

test('combine()', function (t) {
    t.test('both arrays', function (st) {
        var a = [1];
        var b = [2];
        var combined = utils.combine(a, b);

        st.deepEqual(a, [1], 'a is not mutated');
        st.deepEqual(b, [2], 'b is not mutated');
        st.notEqual(a, combined, 'a !== combined');
        st.notEqual(b, combined, 'b !== combined');
        st.deepEqual(combined, [1, 2], 'combined is a + b');

        st.end();
    });

    t.test('one array, one non-array', function (st) {
        var aN = 1;
        var a = [aN];
        var bN = 2;
        var b = [bN];

        var combinedAnB = utils.combine(aN, b);
        st.deepEqual(b, [bN], 'b is not mutated');
        st.notEqual(aN, combinedAnB, 'aN + b !== aN');
        st.notEqual(a, combinedAnB, 'aN + b !== a');
        st.notEqual(bN, combinedAnB, 'aN + b !== bN');
        st.notEqual(b, combinedAnB, 'aN + b !== b');
        st.deepEqual([1, 2], combinedAnB, 'first argument is array-wrapped when not an array');

        var combinedABn = utils.combine(a, bN);
        st.deepEqual(a, [aN], 'a is not mutated');
        st.notEqual(aN, combinedABn, 'a + bN !== aN');
        st.notEqual(a, combinedABn, 'a + bN !== a');
        st.notEqual(bN, combinedABn, 'a + bN !== bN');
        st.notEqual(b, combinedABn, 'a + bN !== b');
        st.deepEqual([1, 2], combinedABn, 'second argument is array-wrapped when not an array');

        st.end();
    });

    t.test('neither is an array', function (st) {
        var combined = utils.combine(1, 2);
        st.notEqual(1, combined, '1 + 2 !== 1');
        st.notEqual(2, combined, '1 + 2 !== 2');
        st.deepEqual([1, 2], combined, 'both arguments are array-wrapped when not an array');

        st.end();
    });

    t.end();
});

test('decode', function (t) {
    t.equal(
        utils.decode('a+b'),
        'a b',
        'decodes + to space'
    );

    t.equal(
        utils.decode('name%2Eobj'),
        'name.obj',
        'decodes a string'
    );
    t.equal(
        utils.decode('name%2Eobj%2Efoo', null, 'iso-8859-1'),
        'name.obj.foo',
        'decodes a string in iso-8859-1'
    );

    t.end();
});

test('encode', function (t) {
    forEach(v.nullPrimitives, function (nullish) {
        t['throws'](
            function () { utils.encode(nullish); },
            TypeError,
            inspect(nullish) + ' is not a string'
        );
    });

    t.equal(utils.encode(''), '', 'empty string returns itself');
    t.deepEqual(utils.encode([]), [], 'empty array returns itself');
    t.deepEqual(utils.encode({ length: 0 }), { length: 0 }, 'empty arraylike returns itself');

    t.test('symbols', { skip: !v.hasSymbols }, function (st) {
        st.equal(utils.encode(Symbol('x')), 'Symbol%28x%29', 'symbol is encoded');

        st.end();
    });

    t.equal(
        utils.encode('(abc)'),
        '%28abc%29',
        'encodes parentheses'
    );
    t.equal(
        utils.encode({ toString: function () { return '(abc)'; } }),
        '%28abc%29',
        'toStrings and encodes parentheses'
    );

    t.equal(
        utils.encode('abc 123 💩', null, 'iso-8859-1'),
        'abc%20123%20%26%2355357%3B%26%2356489%3B',
        'encodes in iso-8859-1'
    );

    var longString = '';
    var expectedString = '';
    for (var i = 0; i < 1500; i++) {
        longString += ' ';
        expectedString += '%20';
    }

    t.equal(
        utils.encode(longString),
        expectedString,
        'encodes a long string'
    );

    t.equal(
        utils.encode('\x28\x29'),
        '%28%29',
        'encodes parens normally'
    );
    t.equal(
        utils.encode('\x28\x29', null, null, null, 'RFC1738'),
        '()',
        'does not encode parens in RFC1738'
    );

    // todo RFC1738 format

    t.equal(
        utils.encode('Āက豈'),
        '%C4%80%E1%80%80%EF%A4%80',
        'encodes multibyte chars'
    );

    t.equal(
        utils.encode('\uD83D \uDCA9'),
        '%F0%9F%90%A0%F0%BA%90%80',
        'encodes lone surrogates'
    );

    t.end();
});

test('isBuffer()', function (t) {
    forEach([null, undefined, true, false, '', 'abc', 42, 0, NaN, {}, [], function () {}, /a/g], function (x) {
        t.equal(utils.isBuffer(x), false, inspect(x) + ' is not a buffer');
    });

    var fakeBuffer = { constructor: Buffer };
    t.equal(utils.isBuffer(fakeBuffer), false, 'fake buffer is not a buffer');

    var saferBuffer = SaferBuffer.from('abc');
    t.equal(utils.isBuffer(saferBuffer), true, 'SaferBuffer instance is a buffer');

    var buffer = Buffer.from && Buffer.alloc ? Buffer.from('abc') : new Buffer('abc');
    t.equal(utils.isBuffer(buffer), true, 'real Buffer instance is a buffer');
    t.end();
});

test('isRegExp()', function (t) {
    t.equal(utils.isRegExp(/a/g), true, 'RegExp is a RegExp');
    t.equal(utils.isRegExp(new RegExp('a', 'g')), true, 'new RegExp is a RegExp');
    t.equal(utils.isRegExp(new Date()), false, 'Date is not a RegExp');

    forEach(v.primitives, function (primitive) {
        t.equal(utils.isRegExp(primitive), false, inspect(primitive) + ' is not a RegExp');
    });

    t.end();
});
