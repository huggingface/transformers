/**
 * The `node:test` module facilitates the creation of JavaScript tests.
 * To access it:
 *
 * ```js
 * import test from 'node:test';
 * ```
 *
 * This module is only available under the `node:` scheme. The following will not
 * work:
 *
 * ```js
 * import test from 'node:test';
 * ```
 *
 * Tests created via the `test` module consist of a single function that is
 * processed in one of three ways:
 *
 * 1. A synchronous function that is considered failing if it throws an exception,
 * and is considered passing otherwise.
 * 2. A function that returns a `Promise` that is considered failing if the `Promise` rejects, and is considered passing if the `Promise` fulfills.
 * 3. A function that receives a callback function. If the callback receives any
 * truthy value as its first argument, the test is considered failing. If a
 * falsy value is passed as the first argument to the callback, the test is
 * considered passing. If the test function receives a callback function and
 * also returns a `Promise`, the test will fail.
 *
 * The following example illustrates how tests are written using the `test` module.
 *
 * ```js
 * test('synchronous passing test', (t) => {
 *   // This test passes because it does not throw an exception.
 *   assert.strictEqual(1, 1);
 * });
 *
 * test('synchronous failing test', (t) => {
 *   // This test fails because it throws an exception.
 *   assert.strictEqual(1, 2);
 * });
 *
 * test('asynchronous passing test', async (t) => {
 *   // This test passes because the Promise returned by the async
 *   // function is settled and not rejected.
 *   assert.strictEqual(1, 1);
 * });
 *
 * test('asynchronous failing test', async (t) => {
 *   // This test fails because the Promise returned by the async
 *   // function is rejected.
 *   assert.strictEqual(1, 2);
 * });
 *
 * test('failing test using Promises', (t) => {
 *   // Promises can be used directly as well.
 *   return new Promise((resolve, reject) => {
 *     setImmediate(() => {
 *       reject(new Error('this will cause the test to fail'));
 *     });
 *   });
 * });
 *
 * test('callback passing test', (t, done) => {
 *   // done() is the callback function. When the setImmediate() runs, it invokes
 *   // done() with no arguments.
 *   setImmediate(done);
 * });
 *
 * test('callback failing test', (t, done) => {
 *   // When the setImmediate() runs, done() is invoked with an Error object and
 *   // the test fails.
 *   setImmediate(() => {
 *     done(new Error('callback failure'));
 *   });
 * });
 * ```
 *
 * If any tests fail, the process exit code is set to `1`.
 * @since v18.0.0, v16.17.0
 * @see [source](https://github.com/nodejs/node/blob/v24.x/lib/test.js)
 */
declare module "node:test" {
    import { Readable } from "node:stream";
    /**
     * **Note:** `shard` is used to horizontally parallelize test running across
     * machines or processes, ideal for large-scale executions across varied
     * environments. It's incompatible with `watch` mode, tailored for rapid
     * code iteration by automatically rerunning tests on file changes.
     *
     * ```js
     * import { tap } from 'node:test/reporters';
     * import { run } from 'node:test';
     * import process from 'node:process';
     * import path from 'node:path';
     *
     * run({ files: [path.resolve('./tests/test.js')] })
     *   .compose(tap)
     *   .pipe(process.stdout);
     * ```
     * @since v18.9.0, v16.19.0
     * @param options Configuration options for running tests.
     */
    function run(options?: RunOptions): TestsStream;
    /**
     * The `test()` function is the value imported from the `test` module. Each
     * invocation of this function results in reporting the test to the `TestsStream`.
     *
     * The `TestContext` object passed to the `fn` argument can be used to perform
     * actions related to the current test. Examples include skipping the test, adding
     * additional diagnostic information, or creating subtests.
     *
     * `test()` returns a `Promise` that fulfills once the test completes.
     * if `test()` is called within a suite, it fulfills immediately.
     * The return value can usually be discarded for top level tests.
     * However, the return value from subtests should be used to prevent the parent
     * test from finishing first and cancelling the subtest
     * as shown in the following example.
     *
     * ```js
     * test('top level test', async (t) => {
     *   // The setTimeout() in the following subtest would cause it to outlive its
     *   // parent test if 'await' is removed on the next line. Once the parent test
     *   // completes, it will cancel any outstanding subtests.
     *   await t.test('longer running subtest', async (t) => {
     *     return new Promise((resolve, reject) => {
     *       setTimeout(resolve, 1000);
     *     });
     *   });
     * });
     * ```
     *
     * The `timeout` option can be used to fail the test if it takes longer than `timeout` milliseconds to complete. However, it is not a reliable mechanism for
     * canceling tests because a running test might block the application thread and
     * thus prevent the scheduled cancellation.
     * @since v18.0.0, v16.17.0
     * @param name The name of the test, which is displayed when reporting test results.
     * Defaults to the `name` property of `fn`, or `'<anonymous>'` if `fn` does not have a name.
     * @param options Configuration options for the test.
     * @param fn The function under test. The first argument to this function is a {@link TestContext} object.
     * If the test uses callbacks, the callback function is passed as the second argument.
     * @return Fulfilled with `undefined` once the test completes, or immediately if the test runs within a suite.
     */
    function test(name?: string, fn?: TestFn): Promise<void>;
    function test(name?: string, options?: TestOptions, fn?: TestFn): Promise<void>;
    function test(options?: TestOptions, fn?: TestFn): Promise<void>;
    function test(fn?: TestFn): Promise<void>;
    namespace test {
        export {
            after,
            afterEach,
            assert,
            before,
            beforeEach,
            describe,
            it,
            mock,
            only,
            run,
            skip,
            snapshot,
            suite,
            test,
            todo,
        };
    }
    /**
     * The `suite()` function is imported from the `node:test` module.
     * @param name The name of the suite, which is displayed when reporting test results.
     * Defaults to the `name` property of `fn`, or `'<anonymous>'` if `fn` does not have a name.
     * @param options Configuration options for the suite. This supports the same options as {@link test}.
     * @param fn The suite function declaring nested tests and suites. The first argument to this function is a {@link SuiteContext} object.
     * @return Immediately fulfilled with `undefined`.
     * @since v20.13.0
     */
    function suite(name?: string, options?: TestOptions, fn?: SuiteFn): Promise<void>;
    function suite(name?: string, fn?: SuiteFn): Promise<void>;
    function suite(options?: TestOptions, fn?: SuiteFn): Promise<void>;
    function suite(fn?: SuiteFn): Promise<void>;
    namespace suite {
        /**
         * Shorthand for skipping a suite. This is the same as calling {@link suite} with `options.skip` set to `true`.
         * @since v20.13.0
         */
        function skip(name?: string, options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function skip(name?: string, fn?: SuiteFn): Promise<void>;
        function skip(options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function skip(fn?: SuiteFn): Promise<void>;
        /**
         * Shorthand for marking a suite as `TODO`. This is the same as calling {@link suite} with `options.todo` set to `true`.
         * @since v20.13.0
         */
        function todo(name?: string, options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function todo(name?: string, fn?: SuiteFn): Promise<void>;
        function todo(options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function todo(fn?: SuiteFn): Promise<void>;
        /**
         * Shorthand for marking a suite as `only`. This is the same as calling {@link suite} with `options.only` set to `true`.
         * @since v20.13.0
         */
        function only(name?: string, options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function only(name?: string, fn?: SuiteFn): Promise<void>;
        function only(options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function only(fn?: SuiteFn): Promise<void>;
    }
    /**
     * Alias for {@link suite}.
     *
     * The `describe()` function is imported from the `node:test` module.
     */
    function describe(name?: string, options?: TestOptions, fn?: SuiteFn): Promise<void>;
    function describe(name?: string, fn?: SuiteFn): Promise<void>;
    function describe(options?: TestOptions, fn?: SuiteFn): Promise<void>;
    function describe(fn?: SuiteFn): Promise<void>;
    namespace describe {
        /**
         * Shorthand for skipping a suite. This is the same as calling {@link describe} with `options.skip` set to `true`.
         * @since v18.15.0
         */
        function skip(name?: string, options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function skip(name?: string, fn?: SuiteFn): Promise<void>;
        function skip(options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function skip(fn?: SuiteFn): Promise<void>;
        /**
         * Shorthand for marking a suite as `TODO`. This is the same as calling {@link describe} with `options.todo` set to `true`.
         * @since v18.15.0
         */
        function todo(name?: string, options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function todo(name?: string, fn?: SuiteFn): Promise<void>;
        function todo(options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function todo(fn?: SuiteFn): Promise<void>;
        /**
         * Shorthand for marking a suite as `only`. This is the same as calling {@link describe} with `options.only` set to `true`.
         * @since v18.15.0
         */
        function only(name?: string, options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function only(name?: string, fn?: SuiteFn): Promise<void>;
        function only(options?: TestOptions, fn?: SuiteFn): Promise<void>;
        function only(fn?: SuiteFn): Promise<void>;
    }
    /**
     * Alias for {@link test}.
     *
     * The `it()` function is imported from the `node:test` module.
     * @since v18.6.0, v16.17.0
     */
    function it(name?: string, options?: TestOptions, fn?: TestFn): Promise<void>;
    function it(name?: string, fn?: TestFn): Promise<void>;
    function it(options?: TestOptions, fn?: TestFn): Promise<void>;
    function it(fn?: TestFn): Promise<void>;
    namespace it {
        /**
         * Shorthand for skipping a test. This is the same as calling {@link it} with `options.skip` set to `true`.
         */
        function skip(name?: string, options?: TestOptions, fn?: TestFn): Promise<void>;
        function skip(name?: string, fn?: TestFn): Promise<void>;
        function skip(options?: TestOptions, fn?: TestFn): Promise<void>;
        function skip(fn?: TestFn): Promise<void>;
        /**
         * Shorthand for marking a test as `TODO`. This is the same as calling {@link it} with `options.todo` set to `true`.
         */
        function todo(name?: string, options?: TestOptions, fn?: TestFn): Promise<void>;
        function todo(name?: string, fn?: TestFn): Promise<void>;
        function todo(options?: TestOptions, fn?: TestFn): Promise<void>;
        function todo(fn?: TestFn): Promise<void>;
        /**
         * Shorthand for marking a test as `only`. This is the same as calling {@link it} with `options.only` set to `true`.
         * @since v18.15.0
         */
        function only(name?: string, options?: TestOptions, fn?: TestFn): Promise<void>;
        function only(name?: string, fn?: TestFn): Promise<void>;
        function only(options?: TestOptions, fn?: TestFn): Promise<void>;
        function only(fn?: TestFn): Promise<void>;
    }
    /**
     * Shorthand for skipping a test. This is the same as calling {@link test} with `options.skip` set to `true`.
     * @since v20.2.0
     */
    function skip(name?: string, options?: TestOptions, fn?: TestFn): Promise<void>;
    function skip(name?: string, fn?: TestFn): Promise<void>;
    function skip(options?: TestOptions, fn?: TestFn): Promise<void>;
    function skip(fn?: TestFn): Promise<void>;
    /**
     * Shorthand for marking a test as `TODO`. This is the same as calling {@link test} with `options.todo` set to `true`.
     * @since v20.2.0
     */
    function todo(name?: string, options?: TestOptions, fn?: TestFn): Promise<void>;
    function todo(name?: string, fn?: TestFn): Promise<void>;
    function todo(options?: TestOptions, fn?: TestFn): Promise<void>;
    function todo(fn?: TestFn): Promise<void>;
    /**
     * Shorthand for marking a test as `only`. This is the same as calling {@link test} with `options.only` set to `true`.
     * @since v20.2.0
     */
    function only(name?: string, options?: TestOptions, fn?: TestFn): Promise<void>;
    function only(name?: string, fn?: TestFn): Promise<void>;
    function only(options?: TestOptions, fn?: TestFn): Promise<void>;
    function only(fn?: TestFn): Promise<void>;
    /**
     * The type of a function passed to {@link test}. The first argument to this function is a {@link TestContext} object.
     * If the test uses callbacks, the callback function is passed as the second argument.
     */
    type TestFn = (t: TestContext, done: (result?: any) => void) => void | Promise<void>;
    /**
     * The type of a suite test function. The argument to this function is a {@link SuiteContext} object.
     */
    type SuiteFn = (s: SuiteContext) => void | Promise<void>;
    interface TestShard {
        /**
         * A positive integer between 1 and `total` that specifies the index of the shard to run.
         */
        index: number;
        /**
         * A positive integer that specifies the total number of shards to split the test files to.
         */
        total: number;
    }
    interface RunOptions {
        /**
         * If a number is provided, then that many test processes would run in parallel, where each process corresponds to one test file.
         * If `true`, it would run `os.availableParallelism() - 1` test files in parallel. If `false`, it would only run one test file at a time.
         * @default false
         */
        concurrency?: number | boolean | undefined;
        /**
         * Specifies the current working directory to be used by the test runner.
         * Serves as the base path for resolving files according to the
         * [test runner execution model](https://nodejs.org/docs/latest-v24.x/api/test.html#test-runner-execution-model).
         * @since v23.0.0
         * @default process.cwd()
         */
        cwd?: string | undefined;
        /**
         * An array containing the list of files to run. If omitted, files are run according to the
         * [test runner execution model](https://nodejs.org/docs/latest-v24.x/api/test.html#test-runner-execution-model).
         */
        files?: readonly string[] | undefined;
        /**
         * Configures the test runner to exit the process once all known
         * tests have finished executing even if the event loop would
         * otherwise remain active.
         * @default false
         */
        forceExit?: boolean | undefined;
        /**
         * An array containing the list of glob patterns to match test files.
         * This option cannot be used together with `files`. If omitted, files are run according to the
         * [test runner execution model](https://nodejs.org/docs/latest-v24.x/api/test.html#test-runner-execution-model).
         * @since v22.6.0
         */
        globPatterns?: readonly string[] | undefined;
        /**
         * Sets inspector port of test child process.
         * This can be a number, or a function that takes no arguments and returns a
         * number. If a nullish value is provided, each process gets its own port,
         * incremented from the primary's `process.debugPort`. This option is ignored
         * if the `isolation` option is set to `'none'` as no child processes are
         * spawned.
         * @default undefined
         */
        inspectPort?: number | (() => number) | undefined;
        /**
         * Configures the type of test isolation. If set to
         * `'process'`, each test file is run in a separate child process. If set to
         * `'none'`, all test files run in the current process.
         * @default 'process'
         * @since v22.8.0
         */
        isolation?: "process" | "none" | undefined;
        /**
         * If truthy, the test context will only run tests that have the `only` option set
         */
        only?: boolean | undefined;
        /**
         * A function that accepts the `TestsStream` instance and can be used to setup listeners before any tests are run.
         * @default undefined
         */
        setup?: ((reporter: TestsStream) => void | Promise<void>) | undefined;
        /**
         * An array of CLI flags to pass to the `node` executable when
         * spawning the subprocesses. This option has no effect when `isolation` is `'none`'.
         * @since v22.10.0
         * @default []
         */
        execArgv?: readonly string[] | undefined;
        /**
         * An array of CLI flags to pass to each test file when spawning the
         * subprocesses. This option has no effect when `isolation` is `'none'`.
         * @since v22.10.0
         * @default []
         */
        argv?: readonly string[] | undefined;
        /**
         * Allows aborting an in-progress test execution.
         */
        signal?: AbortSignal | undefined;
        /**
         * If provided, only run tests whose name matches the provided pattern.
         * Strings are interpreted as JavaScript regular expressions.
         * @default undefined
         */
        testNamePatterns?: string | RegExp | ReadonlyArray<string | RegExp> | undefined;
        /**
         * A String, RegExp or a RegExp Array, that can be used to exclude running tests whose
         * name matches the provided pattern. Test name patterns are interpreted as JavaScript
         * regular expressions. For each test that is executed, any corresponding test hooks,
         * such as `beforeEach()`, are also run.
         * @default undefined
         * @since v22.1.0
         */
        testSkipPatterns?: string | RegExp | ReadonlyArray<string | RegExp> | undefined;
        /**
         * The number of milliseconds after which the test execution will fail.
         * If unspecified, subtests inherit this value from their parent.
         * @default Infinity
         */
        timeout?: number | undefined;
        /**
         * Whether to run in watch mode or not.
         * @default false
         */
        watch?: boolean | undefined;
        /**
         * Running tests in a specific shard.
         * @default undefined
         */
        shard?: TestShard | undefined;
        /**
         * enable [code coverage](https://nodejs.org/docs/latest-v24.x/api/test.html#collecting-code-coverage) collection.
         * @since v22.10.0
         * @default false
         */
        coverage?: boolean | undefined;
        /**
         * Excludes specific files from code coverage
         * using a glob pattern, which can match both absolute and relative file paths.
         * This property is only applicable when `coverage` was set to `true`.
         * If both `coverageExcludeGlobs` and `coverageIncludeGlobs` are provided,
         * files must meet **both** criteria to be included in the coverage report.
         * @since v22.10.0
         * @default undefined
         */
        coverageExcludeGlobs?: string | readonly string[] | undefined;
        /**
         * Includes specific files in code coverage
         * using a glob pattern, which can match both absolute and relative file paths.
         * This property is only applicable when `coverage` was set to `true`.
         * If both `coverageExcludeGlobs` and `coverageIncludeGlobs` are provided,
         * files must meet **both** criteria to be included in the coverage report.
         * @since v22.10.0
         * @default undefined
         */
        coverageIncludeGlobs?: string | readonly string[] | undefined;
        /**
         * Require a minimum percent of covered lines. If code
         * coverage does not reach the threshold specified, the process will exit with code `1`.
         * @since v22.10.0
         * @default 0
         */
        lineCoverage?: number | undefined;
        /**
         * Require a minimum percent of covered branches. If code
         * coverage does not reach the threshold specified, the process will exit with code `1`.
         * @since v22.10.0
         * @default 0
         */
        branchCoverage?: number | undefined;
        /**
         * Require a minimum percent of covered functions. If code
         * coverage does not reach the threshold specified, the process will exit with code `1`.
         * @since v22.10.0
         * @default 0
         */
        functionCoverage?: number | undefined;
    }
    /**
     * A successful call to `run()` will return a new `TestsStream` object, streaming a series of events representing the execution of the tests.
     *
     * Some of the events are guaranteed to be emitted in the same order as the tests are defined, while others are emitted in the order that the tests execute.
     * @since v18.9.0, v16.19.0
     */
    class TestsStream extends Readable implements NodeJS.ReadableStream {
        addListener(event: "test:coverage", listener: (data: TestCoverage) => void): this;
        addListener(event: "test:complete", listener: (data: TestComplete) => void): this;
        addListener(event: "test:dequeue", listener: (data: TestDequeue) => void): this;
        addListener(event: "test:diagnostic", listener: (data: DiagnosticData) => void): this;
        addListener(event: "test:enqueue", listener: (data: TestEnqueue) => void): this;
        addListener(event: "test:fail", listener: (data: TestFail) => void): this;
        addListener(event: "test:pass", listener: (data: TestPass) => void): this;
        addListener(event: "test:plan", listener: (data: TestPlan) => void): this;
        addListener(event: "test:start", listener: (data: TestStart) => void): this;
        addListener(event: "test:stderr", listener: (data: TestStderr) => void): this;
        addListener(event: "test:stdout", listener: (data: TestStdout) => void): this;
        addListener(event: "test:summary", listener: (data: TestSummary) => void): this;
        addListener(event: "test:watch:drained", listener: () => void): this;
        addListener(event: string, listener: (...args: any[]) => void): this;
        emit(event: "test:coverage", data: TestCoverage): boolean;
        emit(event: "test:complete", data: TestComplete): boolean;
        emit(event: "test:dequeue", data: TestDequeue): boolean;
        emit(event: "test:diagnostic", data: DiagnosticData): boolean;
        emit(event: "test:enqueue", data: TestEnqueue): boolean;
        emit(event: "test:fail", data: TestFail): boolean;
        emit(event: "test:pass", data: TestPass): boolean;
        emit(event: "test:plan", data: TestPlan): boolean;
        emit(event: "test:start", data: TestStart): boolean;
        emit(event: "test:stderr", data: TestStderr): boolean;
        emit(event: "test:stdout", data: TestStdout): boolean;
        emit(event: "test:summary", data: TestSummary): boolean;
        emit(event: "test:watch:drained"): boolean;
        emit(event: string | symbol, ...args: any[]): boolean;
        on(event: "test:coverage", listener: (data: TestCoverage) => void): this;
        on(event: "test:complete", listener: (data: TestComplete) => void): this;
        on(event: "test:dequeue", listener: (data: TestDequeue) => void): this;
        on(event: "test:diagnostic", listener: (data: DiagnosticData) => void): this;
        on(event: "test:enqueue", listener: (data: TestEnqueue) => void): this;
        on(event: "test:fail", listener: (data: TestFail) => void): this;
        on(event: "test:pass", listener: (data: TestPass) => void): this;
        on(event: "test:plan", listener: (data: TestPlan) => void): this;
        on(event: "test:start", listener: (data: TestStart) => void): this;
        on(event: "test:stderr", listener: (data: TestStderr) => void): this;
        on(event: "test:stdout", listener: (data: TestStdout) => void): this;
        on(event: "test:summary", listener: (data: TestSummary) => void): this;
        on(event: "test:watch:drained", listener: () => void): this;
        on(event: string, listener: (...args: any[]) => void): this;
        once(event: "test:coverage", listener: (data: TestCoverage) => void): this;
        once(event: "test:complete", listener: (data: TestComplete) => void): this;
        once(event: "test:dequeue", listener: (data: TestDequeue) => void): this;
        once(event: "test:diagnostic", listener: (data: DiagnosticData) => void): this;
        once(event: "test:enqueue", listener: (data: TestEnqueue) => void): this;
        once(event: "test:fail", listener: (data: TestFail) => void): this;
        once(event: "test:pass", listener: (data: TestPass) => void): this;
        once(event: "test:plan", listener: (data: TestPlan) => void): this;
        once(event: "test:start", listener: (data: TestStart) => void): this;
        once(event: "test:stderr", listener: (data: TestStderr) => void): this;
        once(event: "test:stdout", listener: (data: TestStdout) => void): this;
        once(event: "test:summary", listener: (data: TestSummary) => void): this;
        once(event: "test:watch:drained", listener: () => void): this;
        once(event: string, listener: (...args: any[]) => void): this;
        prependListener(event: "test:coverage", listener: (data: TestCoverage) => void): this;
        prependListener(event: "test:complete", listener: (data: TestComplete) => void): this;
        prependListener(event: "test:dequeue", listener: (data: TestDequeue) => void): this;
        prependListener(event: "test:diagnostic", listener: (data: DiagnosticData) => void): this;
        prependListener(event: "test:enqueue", listener: (data: TestEnqueue) => void): this;
        prependListener(event: "test:fail", listener: (data: TestFail) => void): this;
        prependListener(event: "test:pass", listener: (data: TestPass) => void): this;
        prependListener(event: "test:plan", listener: (data: TestPlan) => void): this;
        prependListener(event: "test:start", listener: (data: TestStart) => void): this;
        prependListener(event: "test:stderr", listener: (data: TestStderr) => void): this;
        prependListener(event: "test:stdout", listener: (data: TestStdout) => void): this;
        prependListener(event: "test:summary", listener: (data: TestSummary) => void): this;
        prependListener(event: "test:watch:drained", listener: () => void): this;
        prependListener(event: string, listener: (...args: any[]) => void): this;
        prependOnceListener(event: "test:coverage", listener: (data: TestCoverage) => void): this;
        prependOnceListener(event: "test:complete", listener: (data: TestComplete) => void): this;
        prependOnceListener(event: "test:dequeue", listener: (data: TestDequeue) => void): this;
        prependOnceListener(event: "test:diagnostic", listener: (data: DiagnosticData) => void): this;
        prependOnceListener(event: "test:enqueue", listener: (data: TestEnqueue) => void): this;
        prependOnceListener(event: "test:fail", listener: (data: TestFail) => void): this;
        prependOnceListener(event: "test:pass", listener: (data: TestPass) => void): this;
        prependOnceListener(event: "test:plan", listener: (data: TestPlan) => void): this;
        prependOnceListener(event: "test:start", listener: (data: TestStart) => void): this;
        prependOnceListener(event: "test:stderr", listener: (data: TestStderr) => void): this;
        prependOnceListener(event: "test:stdout", listener: (data: TestStdout) => void): this;
        prependOnceListener(event: "test:summary", listener: (data: TestSummary) => void): this;
        prependOnceListener(event: "test:watch:drained", listener: () => void): this;
        prependOnceListener(event: string, listener: (...args: any[]) => void): this;
    }
    /**
     * An instance of `TestContext` is passed to each test function in order to
     * interact with the test runner. However, the `TestContext` constructor is not
     * exposed as part of the API.
     * @since v18.0.0, v16.17.0
     */
    class TestContext {
        /**
         * An object containing assertion methods bound to the test context.
         * The top-level functions from the `node:assert` module are exposed here for the purpose of creating test plans.
         *
         * **Note:** Some of the functions from `node:assert` contain type assertions. If these are called via the
         * TestContext `assert` object, then the context parameter in the test's function signature **must be explicitly typed**
         * (ie. the parameter must have a type annotation), otherwise an error will be raised by the TypeScript compiler:
         * ```ts
         * import { test, type TestContext } from 'node:test';
         *
         * // The test function's context parameter must have a type annotation.
         * test('example', (t: TestContext) => {
         *   t.assert.deepStrictEqual(actual, expected);
         * });
         *
         * // Omitting the type annotation will result in a compilation error.
         * test('example', t => {
         *   t.assert.deepStrictEqual(actual, expected); // Error: 't' needs an explicit type annotation.
         * });
         * ```
         * @since v22.2.0, v20.15.0
         */
        readonly assert: TestContextAssert;
        /**
         * This function is used to create a hook running before subtest of the current test.
         * @param fn The hook function. The first argument to this function is a `TestContext` object.
         * If the hook uses callbacks, the callback function is passed as the second argument.
         * @param options Configuration options for the hook.
         * @since v20.1.0, v18.17.0
         */
        before(fn?: TestContextHookFn, options?: HookOptions): void;
        /**
         * This function is used to create a hook running before each subtest of the current test.
         * @param fn The hook function. The first argument to this function is a `TestContext` object.
         * If the hook uses callbacks, the callback function is passed as the second argument.
         * @param options Configuration options for the hook.
         * @since v18.8.0
         */
        beforeEach(fn?: TestContextHookFn, options?: HookOptions): void;
        /**
         * This function is used to create a hook that runs after the current test finishes.
         * @param fn The hook function. The first argument to this function is a `TestContext` object.
         * If the hook uses callbacks, the callback function is passed as the second argument.
         * @param options Configuration options for the hook.
         * @since v18.13.0
         */
        after(fn?: TestContextHookFn, options?: HookOptions): void;
        /**
         * This function is used to create a hook running after each subtest of the current test.
         * @param fn The hook function. The first argument to this function is a `TestContext` object.
         * If the hook uses callbacks, the callback function is passed as the second argument.
         * @param options Configuration options for the hook.
         * @since v18.8.0
         */
        afterEach(fn?: TestContextHookFn, options?: HookOptions): void;
        /**
         * This function is used to write diagnostics to the output. Any diagnostic
         * information is included at the end of the test's results. This function does
         * not return a value.
         *
         * ```js
         * test('top level test', (t) => {
         *   t.diagnostic('A diagnostic message');
         * });
         * ```
         * @since v18.0.0, v16.17.0
         * @param message Message to be reported.
         */
        diagnostic(message: string): void;
        /**
         * The absolute path of the test file that created the current test. If a test file imports
         * additional modules that generate tests, the imported tests will return the path of the root test file.
         * @since v22.6.0
         */
        readonly filePath: string | undefined;
        /**
         * The name of the test and each of its ancestors, separated by `>`.
         * @since v22.3.0
         */
        readonly fullName: string;
        /**
         * The name of the test.
         * @since v18.8.0, v16.18.0
         */
        readonly name: string;
        /**
         * This function is used to set the number of assertions and subtests that are expected to run
         * within the test. If the number of assertions and subtests that run does not match the
         * expected count, the test will fail.
         *
         * > Note: To make sure assertions are tracked, `t.assert` must be used instead of `assert` directly.
         *
         * ```js
         * test('top level test', (t) => {
         *   t.plan(2);
         *   t.assert.ok('some relevant assertion here');
         *   t.test('subtest', () => {});
         * });
         * ```
         *
         * When working with asynchronous code, the `plan` function can be used to ensure that the
         * correct number of assertions are run:
         *
         * ```js
         * test('planning with streams', (t, done) => {
         *   function* generate() {
         *     yield 'a';
         *     yield 'b';
         *     yield 'c';
         *   }
         *   const expected = ['a', 'b', 'c'];
         *   t.plan(expected.length);
         *   const stream = Readable.from(generate());
         *   stream.on('data', (chunk) => {
         *     t.assert.strictEqual(chunk, expected.shift());
         *   });
         *
         *   stream.on('end', () => {
         *     done();
         *   });
         * });
         * ```
         *
         * When using the `wait` option, you can control how long the test will wait for the expected assertions.
         * For example, setting a maximum wait time ensures that the test will wait for asynchronous assertions
         * to complete within the specified timeframe:
         *
         * ```js
         * test('plan with wait: 2000 waits for async assertions', (t) => {
         *   t.plan(1, { wait: 2000 }); // Waits for up to 2 seconds for the assertion to complete.
         *
         *   const asyncActivity = () => {
         *     setTimeout(() => {
         *          *       t.assert.ok(true, 'Async assertion completed within the wait time');
         *     }, 1000); // Completes after 1 second, within the 2-second wait time.
         *   };
         *
         *   asyncActivity(); // The test will pass because the assertion is completed in time.
         * });
         * ```
         *
         * Note: If a `wait` timeout is specified, it begins counting down only after the test function finishes executing.
         * @since v22.2.0
         */
        plan(count: number, options?: TestContextPlanOptions): void;
        /**
         * If `shouldRunOnlyTests` is truthy, the test context will only run tests that
         * have the `only` option set. Otherwise, all tests are run. If Node.js was not
         * started with the `--test-only` command-line option, this function is a
         * no-op.
         *
         * ```js
         * test('top level test', (t) => {
         *   // The test context can be set to run subtests with the 'only' option.
         *   t.runOnly(true);
         *   return Promise.all([
         *     t.test('this subtest is now skipped'),
         *     t.test('this subtest is run', { only: true }),
         *   ]);
         * });
         * ```
         * @since v18.0.0, v16.17.0
         * @param shouldRunOnlyTests Whether or not to run `only` tests.
         */
        runOnly(shouldRunOnlyTests: boolean): void;
        /**
         * ```js
         * test('top level test', async (t) => {
         *   await fetch('some/uri', { signal: t.signal });
         * });
         * ```
         * @since v18.7.0, v16.17.0
         */
        readonly signal: AbortSignal;
        /**
         * This function causes the test's output to indicate the test as skipped. If `message` is provided, it is included in the output. Calling `skip()` does
         * not terminate execution of the test function. This function does not return a
         * value.
         *
         * ```js
         * test('top level test', (t) => {
         *   // Make sure to return here as well if the test contains additional logic.
         *   t.skip('this is skipped');
         * });
         * ```
         * @since v18.0.0, v16.17.0
         * @param message Optional skip message.
         */
        skip(message?: string): void;
        /**
         * This function adds a `TODO` directive to the test's output. If `message` is
         * provided, it is included in the output. Calling `todo()` does not terminate
         * execution of the test function. This function does not return a value.
         *
         * ```js
         * test('top level test', (t) => {
         *   // This test is marked as `TODO`
         *   t.todo('this is a todo');
         * });
         * ```
         * @since v18.0.0, v16.17.0
         * @param message Optional `TODO` message.
         */
        todo(message?: string): void;
        /**
         * This function is used to create subtests under the current test. This function behaves in
         * the same fashion as the top level {@link test} function.
         * @since v18.0.0
         * @param name The name of the test, which is displayed when reporting test results.
         * Defaults to the `name` property of `fn`, or `'<anonymous>'` if `fn` does not have a name.
         * @param options Configuration options for the test.
         * @param fn The function under test. This first argument to this function is a {@link TestContext} object.
         * If the test uses callbacks, the callback function is passed as the second argument.
         * @returns A {@link Promise} resolved with `undefined` once the test completes.
         */
        test: typeof test;
        /**
         * This method polls a `condition` function until that function either returns
         * successfully or the operation times out.
         * @since v22.14.0
         * @param condition An assertion function that is invoked
         * periodically until it completes successfully or the defined polling timeout
         * elapses. Successful completion is defined as not throwing or rejecting. This
         * function does not accept any arguments, and is allowed to return any value.
         * @param options An optional configuration object for the polling operation.
         * @returns Fulfilled with the value returned by `condition`.
         */
        waitFor<T>(condition: () => T, options?: TestContextWaitForOptions): Promise<Awaited<T>>;
        /**
         * Each test provides its own MockTracker instance.
         */
        readonly mock: MockTracker;
    }
    interface TestContextAssert extends
        Pick<
            typeof import("assert"),
            | "deepEqual"
            | "deepStrictEqual"
            | "doesNotMatch"
            | "doesNotReject"
            | "doesNotThrow"
            | "equal"
            | "fail"
            | "ifError"
            | "match"
            | "notDeepEqual"
            | "notDeepStrictEqual"
            | "notEqual"
            | "notStrictEqual"
            | "ok"
            | "partialDeepStrictEqual"
            | "rejects"
            | "strictEqual"
            | "throws"
        >
    {
        /**
         * This function serializes `value` and writes it to the file specified by `path`.
         *
         * ```js
         * test('snapshot test with default serialization', (t) => {
         *   t.assert.fileSnapshot({ value1: 1, value2: 2 }, './snapshots/snapshot.json');
         * });
         * ```
         *
         * This function differs from `context.assert.snapshot()` in the following ways:
         *
         * * The snapshot file path is explicitly provided by the user.
         * * Each snapshot file is limited to a single snapshot value.
         * * No additional escaping is performed by the test runner.
         *
         * These differences allow snapshot files to better support features such as syntax
         * highlighting.
         * @since v22.14.0
         * @param value A value to serialize to a string. If Node.js was started with
         * the [`--test-update-snapshots`](https://nodejs.org/docs/latest-v24.x/api/cli.html#--test-update-snapshots)
         * flag, the serialized value is written to
         * `path`. Otherwise, the serialized value is compared to the contents of the
         * existing snapshot file.
         * @param path The file where the serialized `value` is written.
         * @param options Optional configuration options.
         */
        fileSnapshot(value: any, path: string, options?: AssertSnapshotOptions): void;
        /**
         * This function implements assertions for snapshot testing.
         * ```js
         * test('snapshot test with default serialization', (t) => {
         *   t.assert.snapshot({ value1: 1, value2: 2 });
         * });
         *
         * test('snapshot test with custom serialization', (t) => {
         *   t.assert.snapshot({ value3: 3, value4: 4 }, {
         *     serializers: [(value) => JSON.stringify(value)]
         *   });
         * });
         * ```
         * @since v22.3.0
         * @param value A value to serialize to a string. If Node.js was started with
         * the [`--test-update-snapshots`](https://nodejs.org/docs/latest-v24.x/api/cli.html#--test-update-snapshots)
         * flag, the serialized value is written to
         * the snapshot file. Otherwise, the serialized value is compared to the
         * corresponding value in the existing snapshot file.
         */
        snapshot(value: any, options?: AssertSnapshotOptions): void;
        /**
         * A custom assertion function registered with `assert.register()`.
         */
        [name: string]: (...args: any[]) => void;
    }
    interface AssertSnapshotOptions {
        /**
         * An array of synchronous functions used to serialize `value` into a string.
         * `value` is passed as the only argument to the first serializer function.
         * The return value of each serializer is passed as input to the next serializer.
         * Once all serializers have run, the resulting value is coerced to a string.
         *
         * If no serializers are provided, the test runner's default serializers are used.
         */
        serializers?: ReadonlyArray<(value: any) => any> | undefined;
    }
    interface TestContextPlanOptions {
        /**
         * The wait time for the plan:
         * * If `true`, the plan waits indefinitely for all assertions and subtests to run.
         * * If `false`, the plan performs an immediate check after the test function completes,
         * without waiting for any pending assertions or subtests.
         * Any assertions or subtests that complete after this check will not be counted towards the plan.
         * * If a number, it specifies the maximum wait time in milliseconds
         * before timing out while waiting for expected assertions and subtests to be matched.
         * If the timeout is reached, the test will fail.
         * @default false
         */
        wait?: boolean | number | undefined;
    }
    interface TestContextWaitForOptions {
        /**
         * The number of milliseconds to wait after an unsuccessful
         * invocation of `condition` before trying again.
         * @default 50
         */
        interval?: number | undefined;
        /**
         * The poll timeout in milliseconds. If `condition` has not
         * succeeded by the time this elapses, an error occurs.
         * @default 1000
         */
        timeout?: number | undefined;
    }

    /**
     * An instance of `SuiteContext` is passed to each suite function in order to
     * interact with the test runner. However, the `SuiteContext` constructor is not
     * exposed as part of the API.
     * @since v18.7.0, v16.17.0
     */
    class SuiteContext {
        /**
         * The absolute path of the test file that created the current suite. If a test file imports
         * additional modules that generate suites, the imported suites will return the path of the root test file.
         * @since v22.6.0
         */
        readonly filePath: string | undefined;
        /**
         * The name of the suite.
         * @since v18.8.0, v16.18.0
         */
        readonly name: string;
        /**
         * Can be used to abort test subtasks when the test has been aborted.
         * @since v18.7.0, v16.17.0
         */
        readonly signal: AbortSignal;
    }
    interface TestOptions {
        /**
         * If a number is provided, then that many tests would run in parallel.
         * If truthy, it would run (number of cpu cores - 1) tests in parallel.
         * For subtests, it will be `Infinity` tests in parallel.
         * If falsy, it would only run one test at a time.
         * If unspecified, subtests inherit this value from their parent.
         * @default false
         */
        concurrency?: number | boolean | undefined;
        /**
         * If truthy, and the test context is configured to run `only` tests, then this test will be
         * run. Otherwise, the test is skipped.
         * @default false
         */
        only?: boolean | undefined;
        /**
         * Allows aborting an in-progress test.
         * @since v18.8.0
         */
        signal?: AbortSignal | undefined;
        /**
         * If truthy, the test is skipped. If a string is provided, that string is displayed in the
         * test results as the reason for skipping the test.
         * @default false
         */
        skip?: boolean | string | undefined;
        /**
         * A number of milliseconds the test will fail after. If unspecified, subtests inherit this
         * value from their parent.
         * @default Infinity
         * @since v18.7.0
         */
        timeout?: number | undefined;
        /**
         * If truthy, the test marked as `TODO`. If a string is provided, that string is displayed in
         * the test results as the reason why the test is `TODO`.
         * @default false
         */
        todo?: boolean | string | undefined;
        /**
         * The number of assertions and subtests expected to be run in the test.
         * If the number of assertions run in the test does not match the number
         * specified in the plan, the test will fail.
         * @default undefined
         * @since v22.2.0
         */
        plan?: number | undefined;
    }
    /**
     * This function creates a hook that runs before executing a suite.
     *
     * ```js
     * describe('tests', async () => {
     *   before(() => console.log('about to run some test'));
     *   it('is a subtest', () => {
     *     assert.ok('some relevant assertion here');
     *   });
     * });
     * ```
     * @since v18.8.0, v16.18.0
     * @param fn The hook function. If the hook uses callbacks, the callback function is passed as the second argument.
     * @param options Configuration options for the hook.
     */
    function before(fn?: HookFn, options?: HookOptions): void;
    /**
     * This function creates a hook that runs after executing a suite.
     *
     * ```js
     * describe('tests', async () => {
     *   after(() => console.log('finished running tests'));
     *   it('is a subtest', () => {
     *     assert.ok('some relevant assertion here');
     *   });
     * });
     * ```
     * @since v18.8.0, v16.18.0
     * @param fn The hook function. If the hook uses callbacks, the callback function is passed as the second argument.
     * @param options Configuration options for the hook.
     */
    function after(fn?: HookFn, options?: HookOptions): void;
    /**
     * This function creates a hook that runs before each test in the current suite.
     *
     * ```js
     * describe('tests', async () => {
     *   beforeEach(() => console.log('about to run a test'));
     *   it('is a subtest', () => {
     *     assert.ok('some relevant assertion here');
     *   });
     * });
     * ```
     * @since v18.8.0, v16.18.0
     * @param fn The hook function. If the hook uses callbacks, the callback function is passed as the second argument.
     * @param options Configuration options for the hook.
     */
    function beforeEach(fn?: HookFn, options?: HookOptions): void;
    /**
     * This function creates a hook that runs after each test in the current suite.
     * The `afterEach()` hook is run even if the test fails.
     *
     * ```js
     * describe('tests', async () => {
     *   afterEach(() => console.log('finished running a test'));
     *   it('is a subtest', () => {
     *     assert.ok('some relevant assertion here');
     *   });
     * });
     * ```
     * @since v18.8.0, v16.18.0
     * @param fn The hook function. If the hook uses callbacks, the callback function is passed as the second argument.
     * @param options Configuration options for the hook.
     */
    function afterEach(fn?: HookFn, options?: HookOptions): void;
    /**
     * The hook function. The first argument is the context in which the hook is called.
     * If the hook uses callbacks, the callback function is passed as the second argument.
     */
    type HookFn = (c: TestContext | SuiteContext, done: (result?: any) => void) => any;
    /**
     * The hook function. The first argument is a `TestContext` object.
     * If the hook uses callbacks, the callback function is passed as the second argument.
     */
    type TestContextHookFn = (t: TestContext, done: (result?: any) => void) => any;
    /**
     * Configuration options for hooks.
     * @since v18.8.0
     */
    interface HookOptions {
        /**
         * Allows aborting an in-progress hook.
         */
        signal?: AbortSignal | undefined;
        /**
         * A number of milliseconds the hook will fail after. If unspecified, subtests inherit this
         * value from their parent.
         * @default Infinity
         */
        timeout?: number | undefined;
    }
    interface MockFunctionOptions {
        /**
         * The number of times that the mock will use the behavior of `implementation`.
         * Once the mock function has been called `times` times,
         * it will automatically restore the behavior of `original`.
         * This value must be an integer greater than zero.
         * @default Infinity
         */
        times?: number | undefined;
    }
    interface MockMethodOptions extends MockFunctionOptions {
        /**
         * If `true`, `object[methodName]` is treated as a getter.
         * This option cannot be used with the `setter` option.
         */
        getter?: boolean | undefined;
        /**
         * If `true`, `object[methodName]` is treated as a setter.
         * This option cannot be used with the `getter` option.
         */
        setter?: boolean | undefined;
    }
    type Mock<F extends Function> = F & {
        mock: MockFunctionContext<F>;
    };
    type NoOpFunction = (...args: any[]) => undefined;
    type FunctionPropertyNames<T> = {
        [K in keyof T]: T[K] extends Function ? K : never;
    }[keyof T];
    interface MockModuleOptions {
        /**
         * If false, each call to `require()` or `import()` generates a new mock module.
         * If true, subsequent calls will return the same module mock, and the mock module is inserted into the CommonJS cache.
         * @default false
         */
        cache?: boolean | undefined;
        /**
         * The value to use as the mocked module's default export.
         *
         * If this value is not provided, ESM mocks do not include a default export.
         * If the mock is a CommonJS or builtin module, this setting is used as the value of `module.exports`.
         * If this value is not provided, CJS and builtin mocks use an empty object as the value of `module.exports`.
         */
        defaultExport?: any;
        /**
         * An object whose keys and values are used to create the named exports of the mock module.
         *
         * If the mock is a CommonJS or builtin module, these values are copied onto `module.exports`.
         * Therefore, if a mock is created with both named exports and a non-object default export,
         * the mock will throw an exception when used as a CJS or builtin module.
         */
        namedExports?: object | undefined;
    }
    /**
     * The `MockTracker` class is used to manage mocking functionality. The test runner
     * module provides a top level `mock` export which is a `MockTracker` instance.
     * Each test also provides its own `MockTracker` instance via the test context's `mock` property.
     * @since v19.1.0, v18.13.0
     */
    class MockTracker {
        /**
         * This function is used to create a mock function.
         *
         * The following example creates a mock function that increments a counter by one
         * on each invocation. The `times` option is used to modify the mock behavior such
         * that the first two invocations add two to the counter instead of one.
         *
         * ```js
         * test('mocks a counting function', (t) => {
         *   let cnt = 0;
         *
         *   function addOne() {
         *     cnt++;
         *     return cnt;
         *   }
         *
         *   function addTwo() {
         *     cnt += 2;
         *     return cnt;
         *   }
         *
         *   const fn = t.mock.fn(addOne, addTwo, { times: 2 });
         *
         *   assert.strictEqual(fn(), 2);
         *   assert.strictEqual(fn(), 4);
         *   assert.strictEqual(fn(), 5);
         *   assert.strictEqual(fn(), 6);
         * });
         * ```
         * @since v19.1.0, v18.13.0
         * @param original An optional function to create a mock on.
         * @param implementation An optional function used as the mock implementation for `original`. This is useful for creating mocks that exhibit one behavior for a specified number of calls and
         * then restore the behavior of `original`.
         * @param options Optional configuration options for the mock function.
         * @return The mocked function. The mocked function contains a special `mock` property, which is an instance of {@link MockFunctionContext}, and can be used for inspecting and changing the
         * behavior of the mocked function.
         */
        fn<F extends Function = NoOpFunction>(original?: F, options?: MockFunctionOptions): Mock<F>;
        fn<F extends Function = NoOpFunction, Implementation extends Function = F>(
            original?: F,
            implementation?: Implementation,
            options?: MockFunctionOptions,
        ): Mock<F | Implementation>;
        /**
         * This function is used to create a mock on an existing object method. The
         * following example demonstrates how a mock is created on an existing object
         * method.
         *
         * ```js
         * test('spies on an object method', (t) => {
         *   const number = {
         *     value: 5,
         *     subtract(a) {
         *       return this.value - a;
         *     },
         *   };
         *
         *   t.mock.method(number, 'subtract');
         *   assert.strictEqual(number.subtract.mock.calls.length, 0);
         *   assert.strictEqual(number.subtract(3), 2);
         *   assert.strictEqual(number.subtract.mock.calls.length, 1);
         *
         *   const call = number.subtract.mock.calls[0];
         *
         *   assert.deepStrictEqual(call.arguments, [3]);
         *   assert.strictEqual(call.result, 2);
         *   assert.strictEqual(call.error, undefined);
         *   assert.strictEqual(call.target, undefined);
         *   assert.strictEqual(call.this, number);
         * });
         * ```
         * @since v19.1.0, v18.13.0
         * @param object The object whose method is being mocked.
         * @param methodName The identifier of the method on `object` to mock. If `object[methodName]` is not a function, an error is thrown.
         * @param implementation An optional function used as the mock implementation for `object[methodName]`.
         * @param options Optional configuration options for the mock method.
         * @return The mocked method. The mocked method contains a special `mock` property, which is an instance of {@link MockFunctionContext}, and can be used for inspecting and changing the
         * behavior of the mocked method.
         */
        method<
            MockedObject extends object,
            MethodName extends FunctionPropertyNames<MockedObject>,
        >(
            object: MockedObject,
            methodName: MethodName,
            options?: MockFunctionOptions,
        ): MockedObject[MethodName] extends Function ? Mock<MockedObject[MethodName]>
            : never;
        method<
            MockedObject extends object,
            MethodName extends FunctionPropertyNames<MockedObject>,
            Implementation extends Function,
        >(
            object: MockedObject,
            methodName: MethodName,
            implementation: Implementation,
            options?: MockFunctionOptions,
        ): MockedObject[MethodName] extends Function ? Mock<MockedObject[MethodName] | Implementation>
            : never;
        method<MockedObject extends object>(
            object: MockedObject,
            methodName: keyof MockedObject,
            options: MockMethodOptions,
        ): Mock<Function>;
        method<MockedObject extends object>(
            object: MockedObject,
            methodName: keyof MockedObject,
            implementation: Function,
            options: MockMethodOptions,
        ): Mock<Function>;

        /**
         * This function is syntax sugar for `MockTracker.method` with `options.getter` set to `true`.
         * @since v19.3.0, v18.13.0
         */
        getter<
            MockedObject extends object,
            MethodName extends keyof MockedObject,
        >(
            object: MockedObject,
            methodName: MethodName,
            options?: MockFunctionOptions,
        ): Mock<() => MockedObject[MethodName]>;
        getter<
            MockedObject extends object,
            MethodName extends keyof MockedObject,
            Implementation extends Function,
        >(
            object: MockedObject,
            methodName: MethodName,
            implementation?: Implementation,
            options?: MockFunctionOptions,
        ): Mock<(() => MockedObject[MethodName]) | Implementation>;
        /**
         * This function is syntax sugar for `MockTracker.method` with `options.setter` set to `true`.
         * @since v19.3.0, v18.13.0
         */
        setter<
            MockedObject extends object,
            MethodName extends keyof MockedObject,
        >(
            object: MockedObject,
            methodName: MethodName,
            options?: MockFunctionOptions,
        ): Mock<(value: MockedObject[MethodName]) => void>;
        setter<
            MockedObject extends object,
            MethodName extends keyof MockedObject,
            Implementation extends Function,
        >(
            object: MockedObject,
            methodName: MethodName,
            implementation?: Implementation,
            options?: MockFunctionOptions,
        ): Mock<((value: MockedObject[MethodName]) => void) | Implementation>;

        /**
         * This function is used to mock the exports of ECMAScript modules, CommonJS modules, JSON modules, and
         * Node.js builtin modules. Any references to the original module prior to mocking are not impacted. In
         * order to enable module mocking, Node.js must be started with the
         * [`--experimental-test-module-mocks`](https://nodejs.org/docs/latest-v24.x/api/cli.html#--experimental-test-module-mocks)
         * command-line flag.
         *
         * The following example demonstrates how a mock is created for a module.
         *
         * ```js
         * test('mocks a builtin module in both module systems', async (t) => {
         *   // Create a mock of 'node:readline' with a named export named 'fn', which
         *   // does not exist in the original 'node:readline' module.
         *   const mock = t.mock.module('node:readline', {
         *     namedExports: { fn() { return 42; } },
         *   });
         *
         *   let esmImpl = await import('node:readline');
         *   let cjsImpl = require('node:readline');
         *
         *   // cursorTo() is an export of the original 'node:readline' module.
         *   assert.strictEqual(esmImpl.cursorTo, undefined);
         *   assert.strictEqual(cjsImpl.cursorTo, undefined);
         *   assert.strictEqual(esmImpl.fn(), 42);
         *   assert.strictEqual(cjsImpl.fn(), 42);
         *
         *   mock.restore();
         *
         *   // The mock is restored, so the original builtin module is returned.
         *   esmImpl = await import('node:readline');
         *   cjsImpl = require('node:readline');
         *
         *   assert.strictEqual(typeof esmImpl.cursorTo, 'function');
         *   assert.strictEqual(typeof cjsImpl.cursorTo, 'function');
         *   assert.strictEqual(esmImpl.fn, undefined);
         *   assert.strictEqual(cjsImpl.fn, undefined);
         * });
         * ```
         * @since v22.3.0
         * @experimental
         * @param specifier A string identifying the module to mock.
         * @param options Optional configuration options for the mock module.
         */
        module(specifier: string, options?: MockModuleOptions): MockModuleContext;

        /**
         * This function restores the default behavior of all mocks that were previously
         * created by this `MockTracker` and disassociates the mocks from the `MockTracker` instance. Once disassociated, the mocks can still be used, but the `MockTracker` instance can no longer be
         * used to reset their behavior or
         * otherwise interact with them.
         *
         * After each test completes, this function is called on the test context's `MockTracker`. If the global `MockTracker` is used extensively, calling this
         * function manually is recommended.
         * @since v19.1.0, v18.13.0
         */
        reset(): void;
        /**
         * This function restores the default behavior of all mocks that were previously
         * created by this `MockTracker`. Unlike `mock.reset()`, `mock.restoreAll()` does
         * not disassociate the mocks from the `MockTracker` instance.
         * @since v19.1.0, v18.13.0
         */
        restoreAll(): void;

        timers: MockTimers;
    }
    const mock: MockTracker;
    interface MockFunctionCall<
        F extends Function,
        ReturnType = F extends (...args: any) => infer T ? T
            : F extends abstract new(...args: any) => infer T ? T
            : unknown,
        Args = F extends (...args: infer Y) => any ? Y
            : F extends abstract new(...args: infer Y) => any ? Y
            : unknown[],
    > {
        /**
         * An array of the arguments passed to the mock function.
         */
        arguments: Args;
        /**
         * If the mocked function threw then this property contains the thrown value.
         */
        error: unknown | undefined;
        /**
         * The value returned by the mocked function.
         *
         * If the mocked function threw, it will be `undefined`.
         */
        result: ReturnType | undefined;
        /**
         * An `Error` object whose stack can be used to determine the callsite of the mocked function invocation.
         */
        stack: Error;
        /**
         * If the mocked function is a constructor, this field contains the class being constructed.
         * Otherwise this will be `undefined`.
         */
        target: F extends abstract new(...args: any) => any ? F : undefined;
        /**
         * The mocked function's `this` value.
         */
        this: unknown;
    }
    /**
     * The `MockFunctionContext` class is used to inspect or manipulate the behavior of
     * mocks created via the `MockTracker` APIs.
     * @since v19.1.0, v18.13.0
     */
    class MockFunctionContext<F extends Function> {
        /**
         * A getter that returns a copy of the internal array used to track calls to the
         * mock. Each entry in the array is an object with the following properties.
         * @since v19.1.0, v18.13.0
         */
        readonly calls: Array<MockFunctionCall<F>>;
        /**
         * This function returns the number of times that this mock has been invoked. This
         * function is more efficient than checking `ctx.calls.length` because `ctx.calls` is a getter that creates a copy of the internal call tracking array.
         * @since v19.1.0, v18.13.0
         * @return The number of times that this mock has been invoked.
         */
        callCount(): number;
        /**
         * This function is used to change the behavior of an existing mock.
         *
         * The following example creates a mock function using `t.mock.fn()`, calls the
         * mock function, and then changes the mock implementation to a different function.
         *
         * ```js
         * test('changes a mock behavior', (t) => {
         *   let cnt = 0;
         *
         *   function addOne() {
         *     cnt++;
         *     return cnt;
         *   }
         *
         *   function addTwo() {
         *     cnt += 2;
         *     return cnt;
         *   }
         *
         *   const fn = t.mock.fn(addOne);
         *
         *   assert.strictEqual(fn(), 1);
         *   fn.mock.mockImplementation(addTwo);
         *   assert.strictEqual(fn(), 3);
         *   assert.strictEqual(fn(), 5);
         * });
         * ```
         * @since v19.1.0, v18.13.0
         * @param implementation The function to be used as the mock's new implementation.
         */
        mockImplementation(implementation: F): void;
        /**
         * This function is used to change the behavior of an existing mock for a single
         * invocation. Once invocation `onCall` has occurred, the mock will revert to
         * whatever behavior it would have used had `mockImplementationOnce()` not been
         * called.
         *
         * The following example creates a mock function using `t.mock.fn()`, calls the
         * mock function, changes the mock implementation to a different function for the
         * next invocation, and then resumes its previous behavior.
         *
         * ```js
         * test('changes a mock behavior once', (t) => {
         *   let cnt = 0;
         *
         *   function addOne() {
         *     cnt++;
         *     return cnt;
         *   }
         *
         *   function addTwo() {
         *     cnt += 2;
         *     return cnt;
         *   }
         *
         *   const fn = t.mock.fn(addOne);
         *
         *   assert.strictEqual(fn(), 1);
         *   fn.mock.mockImplementationOnce(addTwo);
         *   assert.strictEqual(fn(), 3);
         *   assert.strictEqual(fn(), 4);
         * });
         * ```
         * @since v19.1.0, v18.13.0
         * @param implementation The function to be used as the mock's implementation for the invocation number specified by `onCall`.
         * @param onCall The invocation number that will use `implementation`. If the specified invocation has already occurred then an exception is thrown.
         */
        mockImplementationOnce(implementation: F, onCall?: number): void;
        /**
         * Resets the call history of the mock function.
         * @since v19.3.0, v18.13.0
         */
        resetCalls(): void;
        /**
         * Resets the implementation of the mock function to its original behavior. The
         * mock can still be used after calling this function.
         * @since v19.1.0, v18.13.0
         */
        restore(): void;
    }
    /**
     * @since v22.3.0
     * @experimental
     */
    class MockModuleContext {
        /**
         * Resets the implementation of the mock module.
         * @since v22.3.0
         */
        restore(): void;
    }

    type Timer = "setInterval" | "setTimeout" | "setImmediate" | "Date";
    interface MockTimersOptions {
        apis: Timer[];
        now?: number | Date | undefined;
    }
    /**
     * Mocking timers is a technique commonly used in software testing to simulate and
     * control the behavior of timers, such as `setInterval` and `setTimeout`,
     * without actually waiting for the specified time intervals.
     *
     * The MockTimers API also allows for mocking of the `Date` constructor and
     * `setImmediate`/`clearImmediate` functions.
     *
     * The `MockTracker` provides a top-level `timers` export
     * which is a `MockTimers` instance.
     * @since v20.4.0
     */
    class MockTimers {
        /**
         * Enables timer mocking for the specified timers.
         *
         * **Note:** When you enable mocking for a specific timer, its associated
         * clear function will also be implicitly mocked.
         *
         * **Note:** Mocking `Date` will affect the behavior of the mocked timers
         * as they use the same internal clock.
         *
         * Example usage without setting initial time:
         *
         * ```js
         * import { mock } from 'node:test';
         * mock.timers.enable({ apis: ['setInterval', 'Date'], now: 1234 });
         * ```
         *
         * The above example enables mocking for the `Date` constructor, `setInterval` timer and
         * implicitly mocks the `clearInterval` function. Only the `Date` constructor from `globalThis`,
         * `setInterval` and `clearInterval` functions from `node:timers`, `node:timers/promises`, and `globalThis` will be mocked.
         *
         * Example usage with initial time set
         *
         * ```js
         * import { mock } from 'node:test';
         * mock.timers.enable({ apis: ['Date'], now: 1000 });
         * ```
         *
         * Example usage with initial Date object as time set
         *
         * ```js
         * import { mock } from 'node:test';
         * mock.timers.enable({ apis: ['Date'], now: new Date() });
         * ```
         *
         * Alternatively, if you call `mock.timers.enable()` without any parameters:
         *
         * All timers (`'setInterval'`, `'clearInterval'`, `'Date'`, `'setImmediate'`, `'clearImmediate'`, `'setTimeout'`, and `'clearTimeout'`)
         * will be mocked.
         *
         * The `setInterval`, `clearInterval`, `setTimeout`, and `clearTimeout` functions from `node:timers`, `node:timers/promises`,
         * and `globalThis` will be mocked.
         * The `Date` constructor from `globalThis` will be mocked.
         *
         * If there is no initial epoch set, the initial date will be based on 0 in the Unix epoch. This is `January 1st, 1970, 00:00:00 UTC`. You can
         * set an initial date by passing a now property to the `.enable()` method. This value will be used as the initial date for the mocked Date
         * object. It can either be a positive integer, or another Date object.
         * @since v20.4.0
         */
        enable(options?: MockTimersOptions): void;
        /**
         * You can use the `.setTime()` method to manually move the mocked date to another time. This method only accepts a positive integer.
         * Note: This method will execute any mocked timers that are in the past from the new time.
         * In the below example we are setting a new time for the mocked date.
         * ```js
         * import assert from 'node:assert';
         * import { test } from 'node:test';
         * test('sets the time of a date object', (context) => {
         *   // Optionally choose what to mock
         *   context.mock.timers.enable({ apis: ['Date'], now: 100 });
         *   assert.strictEqual(Date.now(), 100);
         *   // Advance in time will also advance the date
         *   context.mock.timers.setTime(1000);
         *   context.mock.timers.tick(200);
         *   assert.strictEqual(Date.now(), 1200);
         * });
         * ```
         */
        setTime(time: number): void;
        /**
         * This function restores the default behavior of all mocks that were previously
         * created by this `MockTimers` instance and disassociates the mocks
         * from the `MockTracker` instance.
         *
         * **Note:** After each test completes, this function is called on
         * the test context's `MockTracker`.
         *
         * ```js
         * import { mock } from 'node:test';
         * mock.timers.reset();
         * ```
         * @since v20.4.0
         */
        reset(): void;
        /**
         * Advances time for all mocked timers.
         *
         * **Note:** This diverges from how `setTimeout` in Node.js behaves and accepts
         * only positive numbers. In Node.js, `setTimeout` with negative numbers is
         * only supported for web compatibility reasons.
         *
         * The following example mocks a `setTimeout` function and
         * by using `.tick` advances in
         * time triggering all pending timers.
         *
         * ```js
         * import assert from 'node:assert';
         * import { test } from 'node:test';
         *
         * test('mocks setTimeout to be executed synchronously without having to actually wait for it', (context) => {
         *   const fn = context.mock.fn();
         *
         *   context.mock.timers.enable({ apis: ['setTimeout'] });
         *
         *   setTimeout(fn, 9999);
         *
         *   assert.strictEqual(fn.mock.callCount(), 0);
         *
         *   // Advance in time
         *   context.mock.timers.tick(9999);
         *
         *   assert.strictEqual(fn.mock.callCount(), 1);
         * });
         * ```
         *
         * Alternativelly, the `.tick` function can be called many times
         *
         * ```js
         * import assert from 'node:assert';
         * import { test } from 'node:test';
         *
         * test('mocks setTimeout to be executed synchronously without having to actually wait for it', (context) => {
         *   const fn = context.mock.fn();
         *   context.mock.timers.enable({ apis: ['setTimeout'] });
         *   const nineSecs = 9000;
         *   setTimeout(fn, nineSecs);
         *
         *   const twoSeconds = 3000;
         *   context.mock.timers.tick(twoSeconds);
         *   context.mock.timers.tick(twoSeconds);
         *   context.mock.timers.tick(twoSeconds);
         *
         *   assert.strictEqual(fn.mock.callCount(), 1);
         * });
         * ```
         *
         * Advancing time using `.tick` will also advance the time for any `Date` object
         * created after the mock was enabled (if `Date` was also set to be mocked).
         *
         * ```js
         * import assert from 'node:assert';
         * import { test } from 'node:test';
         *
         * test('mocks setTimeout to be executed synchronously without having to actually wait for it', (context) => {
         *   const fn = context.mock.fn();
         *
         *   context.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
         *   setTimeout(fn, 9999);
         *
         *   assert.strictEqual(fn.mock.callCount(), 0);
         *   assert.strictEqual(Date.now(), 0);
         *
         *   // Advance in time
         *   context.mock.timers.tick(9999);
         *   assert.strictEqual(fn.mock.callCount(), 1);
         *   assert.strictEqual(Date.now(), 9999);
         * });
         * ```
         * @since v20.4.0
         */
        tick(milliseconds: number): void;
        /**
         * Triggers all pending mocked timers immediately. If the `Date` object is also
         * mocked, it will also advance the `Date` object to the furthest timer's time.
         *
         * The example below triggers all pending timers immediately,
         * causing them to execute without any delay.
         *
         * ```js
         * import assert from 'node:assert';
         * import { test } from 'node:test';
         *
         * test('runAll functions following the given order', (context) => {
         *   context.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
         *   const results = [];
         *   setTimeout(() => results.push(1), 9999);
         *
         *   // Notice that if both timers have the same timeout,
         *   // the order of execution is guaranteed
         *   setTimeout(() => results.push(3), 8888);
         *   setTimeout(() => results.push(2), 8888);
         *
         *   assert.deepStrictEqual(results, []);
         *
         *   context.mock.timers.runAll();
         *   assert.deepStrictEqual(results, [3, 2, 1]);
         *   // The Date object is also advanced to the furthest timer's time
         *   assert.strictEqual(Date.now(), 9999);
         * });
         * ```
         *
         * **Note:** The `runAll()` function is specifically designed for
         * triggering timers in the context of timer mocking.
         * It does not have any effect on real-time system
         * clocks or actual timers outside of the mocking environment.
         * @since v20.4.0
         */
        runAll(): void;
        /**
         * Calls {@link MockTimers.reset()}.
         */
        [Symbol.dispose](): void;
    }
    /**
     * An object whose methods are used to configure available assertions on the
     * `TestContext` objects in the current process. The methods from `node:assert`
     * and snapshot testing functions are available by default.
     *
     * It is possible to apply the same configuration to all files by placing common
     * configuration code in a module
     * preloaded with `--require` or `--import`.
     * @since v22.14.0
     */
    namespace assert {
        /**
         * Defines a new assertion function with the provided name and function. If an
         * assertion already exists with the same name, it is overwritten.
         * @since v22.14.0
         */
        function register(name: string, fn: (this: TestContext, ...args: any[]) => void): void;
    }
    /**
     * @since v22.3.0
     */
    namespace snapshot {
        /**
         * This function is used to customize the default serialization mechanism used by the test runner.
         *
         * By default, the test runner performs serialization by calling `JSON.stringify(value, null, 2)` on the provided value.
         * `JSON.stringify()` does have limitations regarding circular structures and supported data types.
         * If a more robust serialization mechanism is required, this function should be used to specify a list of custom serializers.
         *
         * Serializers are called in order, with the output of the previous serializer passed as input to the next.
         * The final result must be a string value.
         * @since v22.3.0
         * @param serializers An array of synchronous functions used as the default serializers for snapshot tests.
         */
        function setDefaultSnapshotSerializers(serializers: ReadonlyArray<(value: any) => any>): void;
        /**
         * This function is used to set a custom resolver for the location of the snapshot file used for snapshot testing.
         * By default, the snapshot filename is the same as the entry point filename with `.snapshot` appended.
         * @since v22.3.0
         * @param fn A function used to compute the location of the snapshot file.
         * The function receives the path of the test file as its only argument. If the
         * test is not associated with a file (for example in the REPL), the input is
         * undefined. `fn()` must return a string specifying the location of the snapshot file.
         */
        function setResolveSnapshotPath(fn: (path: string | undefined) => string): void;
    }
    export {
        after,
        afterEach,
        assert,
        before,
        beforeEach,
        describe,
        it,
        Mock,
        mock,
        only,
        run,
        skip,
        snapshot,
        suite,
        SuiteContext,
        test,
        test as default,
        TestContext,
        todo,
    };
}

interface TestError extends Error {
    cause: Error;
}
interface TestLocationInfo {
    /**
     * The column number where the test is defined, or
     * `undefined` if the test was run through the REPL.
     */
    column?: number;
    /**
     * The path of the test file, `undefined` if test was run through the REPL.
     */
    file?: string;
    /**
     * The line number where the test is defined, or `undefined` if the test was run through the REPL.
     */
    line?: number;
}
interface DiagnosticData extends TestLocationInfo {
    /**
     * The diagnostic message.
     */
    message: string;
    /**
     * The nesting level of the test.
     */
    nesting: number;
}
interface TestCoverage {
    /**
     * An object containing the coverage report.
     */
    summary: {
        /**
         * An array of coverage reports for individual files.
         */
        files: Array<{
            /**
             * The absolute path of the file.
             */
            path: string;
            /**
             * The total number of lines.
             */
            totalLineCount: number;
            /**
             * The total number of branches.
             */
            totalBranchCount: number;
            /**
             * The total number of functions.
             */
            totalFunctionCount: number;
            /**
             * The number of covered lines.
             */
            coveredLineCount: number;
            /**
             * The number of covered branches.
             */
            coveredBranchCount: number;
            /**
             * The number of covered functions.
             */
            coveredFunctionCount: number;
            /**
             * The percentage of lines covered.
             */
            coveredLinePercent: number;
            /**
             * The percentage of branches covered.
             */
            coveredBranchPercent: number;
            /**
             * The percentage of functions covered.
             */
            coveredFunctionPercent: number;
            /**
             * An array of functions representing function coverage.
             */
            functions: Array<{
                /**
                 * The name of the function.
                 */
                name: string;
                /**
                 * The line number where the function is defined.
                 */
                line: number;
                /**
                 * The number of times the function was called.
                 */
                count: number;
            }>;
            /**
             * An array of branches representing branch coverage.
             */
            branches: Array<{
                /**
                 * The line number where the branch is defined.
                 */
                line: number;
                /**
                 * The number of times the branch was taken.
                 */
                count: number;
            }>;
            /**
             * An array of lines representing line numbers and the number of times they were covered.
             */
            lines: Array<{
                /**
                 * The line number.
                 */
                line: number;
                /**
                 * The number of times the line was covered.
                 */
                count: number;
            }>;
        }>;
        /**
         * An object containing whether or not the coverage for
         * each coverage type.
         * @since v22.9.0
         */
        thresholds: {
            /**
             * The function coverage threshold.
             */
            function: number;
            /**
             * The branch coverage threshold.
             */
            branch: number;
            /**
             * The line coverage threshold.
             */
            line: number;
        };
        /**
         * An object containing a summary of coverage for all files.
         */
        totals: {
            /**
             * The total number of lines.
             */
            totalLineCount: number;
            /**
             * The total number of branches.
             */
            totalBranchCount: number;
            /**
             * The total number of functions.
             */
            totalFunctionCount: number;
            /**
             * The number of covered lines.
             */
            coveredLineCount: number;
            /**
             * The number of covered branches.
             */
            coveredBranchCount: number;
            /**
             * The number of covered functions.
             */
            coveredFunctionCount: number;
            /**
             * The percentage of lines covered.
             */
            coveredLinePercent: number;
            /**
             * The percentage of branches covered.
             */
            coveredBranchPercent: number;
            /**
             * The percentage of functions covered.
             */
            coveredFunctionPercent: number;
        };
        /**
         * The working directory when code coverage began. This
         * is useful for displaying relative path names in case
         * the tests changed the working directory of the Node.js process.
         */
        workingDirectory: string;
    };
    /**
     * The nesting level of the test.
     */
    nesting: number;
}
interface TestComplete extends TestLocationInfo {
    /**
     * Additional execution metadata.
     */
    details: {
        /**
         * Whether the test passed or not.
         */
        passed: boolean;
        /**
         * The duration of the test in milliseconds.
         */
        duration_ms: number;
        /**
         * An error wrapping the error thrown by the test if it did not pass.
         */
        error?: TestError;
        /**
         * The type of the test, used to denote whether this is a suite.
         */
        type?: "suite";
    };
    /**
     * The test name.
     */
    name: string;
    /**
     * The nesting level of the test.
     */
    nesting: number;
    /**
     * The ordinal number of the test.
     */
    testNumber: number;
    /**
     * Present if `context.todo` is called.
     */
    todo?: string | boolean;
    /**
     * Present if `context.skip` is called.
     */
    skip?: string | boolean;
}
interface TestDequeue extends TestLocationInfo {
    /**
     * The test name.
     */
    name: string;
    /**
     * The nesting level of the test.
     */
    nesting: number;
    /**
     * The test type. Either `'suite'` or `'test'`.
     * @since v22.15.0
     */
    type: "suite" | "test";
}
interface TestEnqueue extends TestLocationInfo {
    /**
     * The test name.
     */
    name: string;
    /**
     * The nesting level of the test.
     */
    nesting: number;
    /**
     * The test type. Either `'suite'` or `'test'`.
     * @since v22.15.0
     */
    type: "suite" | "test";
}
interface TestFail extends TestLocationInfo {
    /**
     * Additional execution metadata.
     */
    details: {
        /**
         * The duration of the test in milliseconds.
         */
        duration_ms: number;
        /**
         * An error wrapping the error thrown by the test.
         */
        error: TestError;
        /**
         * The type of the test, used to denote whether this is a suite.
         * @since v20.0.0, v19.9.0, v18.17.0
         */
        type?: "suite";
    };
    /**
     * The test name.
     */
    name: string;
    /**
     * The nesting level of the test.
     */
    nesting: number;
    /**
     * The ordinal number of the test.
     */
    testNumber: number;
    /**
     * Present if `context.todo` is called.
     */
    todo?: string | boolean;
    /**
     * Present if `context.skip` is called.
     */
    skip?: string | boolean;
}
interface TestPass extends TestLocationInfo {
    /**
     * Additional execution metadata.
     */
    details: {
        /**
         * The duration of the test in milliseconds.
         */
        duration_ms: number;
        /**
         * The type of the test, used to denote whether this is a suite.
         * @since 20.0.0, 19.9.0, 18.17.0
         */
        type?: "suite";
    };
    /**
     * The test name.
     */
    name: string;
    /**
     * The nesting level of the test.
     */
    nesting: number;
    /**
     * The ordinal number of the test.
     */
    testNumber: number;
    /**
     * Present if `context.todo` is called.
     */
    todo?: string | boolean;
    /**
     * Present if `context.skip` is called.
     */
    skip?: string | boolean;
}
interface TestPlan extends TestLocationInfo {
    /**
     * The nesting level of the test.
     */
    nesting: number;
    /**
     * The number of subtests that have ran.
     */
    count: number;
}
interface TestStart extends TestLocationInfo {
    /**
     * The test name.
     */
    name: string;
    /**
     * The nesting level of the test.
     */
    nesting: number;
}
interface TestStderr {
    /**
     * The path of the test file.
     */
    file: string;
    /**
     * The message written to `stderr`.
     */
    message: string;
}
interface TestStdout {
    /**
     * The path of the test file.
     */
    file: string;
    /**
     * The message written to `stdout`.
     */
    message: string;
}
interface TestSummary {
    /**
     * An object containing the counts of various test results.
     */
    counts: {
        /**
         * The total number of cancelled tests.
         */
        cancelled: number;
        /**
         * The total number of passed tests.
         */
        passed: number;
        /**
         * The total number of skipped tests.
         */
        skipped: number;
        /**
         * The total number of suites run.
         */
        suites: number;
        /**
         * The total number of tests run, excluding suites.
         */
        tests: number;
        /**
         * The total number of TODO tests.
         */
        todo: number;
        /**
         * The total number of top level tests and suites.
         */
        topLevel: number;
    };
    /**
     * The duration of the test run in milliseconds.
     */
    duration_ms: number;
    /**
     * The path of the test file that generated the
     * summary. If the summary corresponds to multiple files, this value is
     * `undefined`.
     */
    file: string | undefined;
    /**
     * Indicates whether or not the test run is considered
     * successful or not. If any error condition occurs, such as a failing test or
     * unmet coverage threshold, this value will be set to `false`.
     */
    success: boolean;
}

/**
 * The `node:test/reporters` module exposes the builtin-reporters for `node:test`.
 * To access it:
 *
 * ```js
 * import test from 'node:test/reporters';
 * ```
 *
 * This module is only available under the `node:` scheme. The following will not
 * work:
 *
 * ```js
 * import test from 'node:test/reporters';
 * ```
 * @since v19.9.0
 * @see [source](https://github.com/nodejs/node/blob/v24.x/lib/test/reporters.js)
 */
declare module "node:test/reporters" {
    import { Transform, TransformOptions } from "node:stream";

    type TestEvent =
        | { type: "test:coverage"; data: TestCoverage }
        | { type: "test:complete"; data: TestComplete }
        | { type: "test:dequeue"; data: TestDequeue }
        | { type: "test:diagnostic"; data: DiagnosticData }
        | { type: "test:enqueue"; data: TestEnqueue }
        | { type: "test:fail"; data: TestFail }
        | { type: "test:pass"; data: TestPass }
        | { type: "test:plan"; data: TestPlan }
        | { type: "test:start"; data: TestStart }
        | { type: "test:stderr"; data: TestStderr }
        | { type: "test:stdout"; data: TestStdout }
        | { type: "test:summary"; data: TestSummary }
        | { type: "test:watch:drained"; data: undefined };
    type TestEventGenerator = AsyncGenerator<TestEvent, void>;

    interface ReporterConstructorWrapper<T extends new(...args: any[]) => Transform> {
        new(...args: ConstructorParameters<T>): InstanceType<T>;
        (...args: ConstructorParameters<T>): InstanceType<T>;
    }

    /**
     * The `dot` reporter outputs the test results in a compact format,
     * where each passing test is represented by a `.`,
     * and each failing test is represented by a `X`.
     * @since v20.0.0
     */
    function dot(source: TestEventGenerator): AsyncGenerator<"\n" | "." | "X", void>;
    /**
     * The `tap` reporter outputs the test results in the [TAP](https://testanything.org/) format.
     * @since v20.0.0
     */
    function tap(source: TestEventGenerator): AsyncGenerator<string, void>;
    class SpecReporter extends Transform {
        constructor();
    }
    /**
     * The `spec` reporter outputs the test results in a human-readable format.
     * @since v20.0.0
     */
    const spec: ReporterConstructorWrapper<typeof SpecReporter>;
    /**
     * The `junit` reporter outputs test results in a jUnit XML format.
     * @since v21.0.0
     */
    function junit(source: TestEventGenerator): AsyncGenerator<string, void>;
    class LcovReporter extends Transform {
        constructor(opts?: Omit<TransformOptions, "writableObjectMode">);
    }
    /**
     * The `lcov` reporter outputs test coverage when used with the
     * [`--experimental-test-coverage`](https://nodejs.org/docs/latest-v24.x/api/cli.html#--experimental-test-coverage) flag.
     * @since v22.0.0
     */
    const lcov: ReporterConstructorWrapper<typeof LcovReporter>;

    export { dot, junit, lcov, spec, tap, TestEvent };
}
