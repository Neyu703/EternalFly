import { setConsoleFunction } from "three";

// @react-three/fiber (through 9.8.1) still creates a THREE.Clock for every canvas, which
// three.js deprecated in r183; the warning names library internals this app can't change.
const FIBER_CLOCK_DEPRECATION = "THREE.Clock: This module has been deprecated.";

/** A three.js stack-trace parameter, which turns the message into an Error with that trace. */
type ThreeStackTrace = { isStackTrace: true; getError: (message: string) => Error };

/** Whether a message parameter is such a stack trace. */
function isThreeStackTrace(value: unknown): value is ThreeStackTrace {
    return typeof value === "object" && value !== null && "isStackTrace" in value && value.isStackTrace === true;
}

/** Routes three.js's console output to the browser console as three.js itself would,
 * minus the one deprecation warning that comes from @react-three/fiber, not from this app. */
export function routeThreeConsoleOutput(): void {
    setConsoleFunction((type, message, ...params) => {
        if (message.startsWith(FIBER_CLOCK_DEPRECATION)) return;
        const [stackTrace] = params;
        if (isThreeStackTrace(stackTrace)) {
            console[type](stackTrace.getError(message));
        } else {
            console[type](message, ...params);
        }
    });
}
