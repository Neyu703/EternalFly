/** A copy of items with item appended, keeping only the last maxLength entries. */
export function appendCapped<Item>(items: Item[], item: Item, maxLength: number): Item[] {
    const nextItems = [...items, item];
    return nextItems.length > maxLength ? nextItems.slice(-maxLength) : nextItems;
}
